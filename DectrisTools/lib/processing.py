"""
module for data processing tools
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from collections.abc import Iterable
import warnings
from os import path, listdir, remove
from pathlib import Path
import re
from datetime import datetime
from psutil import virtual_memory
import numpy as np
import hdf5plugin
import h5py
from tqdm import tqdm
from numba import jit, prange
from .computation import masked_sum


@jit(nopython=True)
def indexed_masked_sum(images, slices, mask):
    n_imgs = images.shape[0]
    ret = np.empty(n_imgs)
    for i in prange(n_imgs):
        ret[i] = np.sum(images[i, slices[0], slices[1]] * mask[slices[0], slices[1]])
    return ret


@jit(nopython=True)
def masked_ravel(images, mask):
    n_imgs = images.shape[0]
    n_pix = np.sum(mask)
    ret = np.zeros(n_pix * n_imgs)
    for i in prange(n_imgs):
        ret[i * n_pix:(i + 1) * n_pix] = (images[i] * mask).ravel()
    return ret


@jit(nopython=True)
def normed_sum(images, norm_values):
    n_imgs = images.shape[0]
    ret = np.zeros(images[0].shape)
    for i in range(n_imgs):
        ret += images[i] / norm_values[i]
    return ret


@jit(nopython=True)
def masked_histogram(images, mask):
    bins = np.zeros(2**16, dtype=np.uint64)
    n_imgs = images.shape[0]
    n_pix_masked = (mask.shape[0] * mask.shape[1]) - np.sum(mask)
    for i in range(n_imgs):
        image = images[i] * mask
        for j in range(images.shape[1]):
            for k in range(images.shape[2]):
                bins[image[j, k]] += 1
    bins[0] -= n_imgs * n_pix_masked  # subtract masked pixels that have been counted
    return bins


class AlreadyProcessedWarning(Warning):
    pass


class UndistinguishableWarning(Warning):
    pass


class BrokenImageWarning(Warning):
    pass


class SingleShotProcessor(ThreadPoolExecutor):
    """class to process hdf5 files from single shot experiments
    each hdf5 file will contain a series of images that are contain pump on/off data
    we use the pump on laser reflections at the detector border to determine which images are which and subsequently
    a new hdf5 file called `{old_filename}_processed.h5` with the structure, where N is the total number of images in
    the file to be processed:
    /                         Group
    /confidence               Dataset {SCALAR} -> confidence that pump on/off is correctly identified
    /difference               Dataset {image_x, image_y} -> mean difference image
    /pump_off                 Dataset {image_x, image_y} -> mean pump off image
    /pump_off_sum_intensities Dataset {N/2} -> sum intensity of every pump off image
    /pump_on                  Dataset {image_x, image_y} -> mean pump on image
    /pump_on_sum_intensities  Dataset {N/2} -> sum intensity of every pump on image

    Example:
    >>>from pathlib import Path
    >>>from concurrent.futures import as_completed

    >>>rundir = "/data/TiSe2_run_0010"
    >>>with SingleShotProcessor([str(p) for p in Path(rundir).rglob("*.h5")], max_workers=1) as ssp:
    >>>    ssp.start()
    >>>    for future in as_completed(ssp.futures):
    >>>        print(ssp[future])
    """

    BORDERSIZE = 8

    def __init__(
        self,
        filelist,
        mask=None,
        max_workers=None,
        ignore_existing=False,
        discard_fist_last_img=True,
    ):
        self.filelist = filelist
        self.ignore_existing = ignore_existing
        self.discard_fist_last_img = discard_fist_last_img
        self.sample_dataset = h5py.File(filelist[0], "r")["entry/data/data"]
        if max_workers is None:
            # determine max workers from free memory and size of dataset
            free_memory = virtual_memory().available
            datasets_in_memory = int(
                free_memory
                / (
                    self.sample_dataset.dtype.itemsize
                    * np.prod(self.sample_dataset.shape)
                )
            )
            if datasets_in_memory == 0:
                warnings.warn(
                    "you might want to free up some system memory; you can't fit a whole dataset into it",
                    ResourceWarning,
                )
                datasets_in_memory = 1
            max_workers = datasets_in_memory
        self.img_size = self.sample_dataset.shape[1:]
        if mask is None:
            self.mask = np.ones(self.img_size).astype(np.uint16)
        else:
            self.mask = mask.astype(np.uint16)
        self.n_imgs = self.sample_dataset.shape[0]
        self.border_mask = np.ones(self.img_size).astype(np.uint16)
        self.border_mask[
            self.BORDERSIZE : -self.BORDERSIZE, self.BORDERSIZE : -self.BORDERSIZE
        ] = 0
        self.futures = {}
        super().__init__(max_workers=max_workers)

    def __getitem__(self, key):
        return self.futures[key]

    def watch(self):
        Thread(target=self.__watch).start()

    def __watch(self):
        for future in as_completed(self.futures):
            if future.exception():
                if isinstance(future.exception(), OSError):
                    warnings.warn(str(future.exception()), AlreadyProcessedWarning)
                else:
                    raise future.exception()

    def submit(self, filename):
        return super().submit(self.__worker, filename)

    def start(self):
        self.futures = {self.submit(fname): fname for fname in self.filelist}
        self.watch()

    def shutdown(self, *args, **kwargs):
        super().shutdown(*args, **kwargs)

    def __worker(self, filename):
        dirname = path.dirname(filename)
        processed_filename = f"{path.join(dirname, Path(filename).stem)}_processed.h5"
        if path.exists(processed_filename):
            if not self.ignore_existing:
                raise OSError(f"{processed_filename} already exists")
        if "pumpon" in filename:
            self.__process_pump_probe(filename, processed_filename)
        elif "pump_off" in filename:
            self.__process_diagnostics(filename, processed_filename, "pump_off")
        elif "laser_bg" in filename:
            self.__process_diagnostics(filename, processed_filename, "laser_bg")
        else:
            raise NotImplementedError(f"don't know what to do with {filename}")

    def __process_diagnostics(self, src, dest, name):
        with h5py.File(src, "r") as f:
            if self.discard_fist_last_img:
                images = f["entry/data/data"][()][1:-1]
            else:
                images = f["entry/data/data"][()]
            # in rare cases we observe broken images that show vertical stripes with values of 2**16-1 = 65535
            # if we find an image like that, we just drop the entire batch of images
            # specifically we check the 150th column of all the images and check how often we find the value 65535
            # if it occurs more than 3 times, we drop the file
            if np.sum((images[:, :, 150] * self.mask[:, 150]) == 65535) > 3:
                warnings.warn(f"found broken image in: {src}; skipping...")
                return

            # look at the intensity sum in the first images and compare them; darks will be dark...
            if self.n_imgs > 100:
                sum_1 = np.sum(images[:100:2])
                sum_2 = np.sum(images[1:101:2])
            else:
                sum_1 = np.sum(images[::2])
                sum_2 = np.sum(images[1::2])
            if sum_1 > sum_2:
                data_mean = np.mean(images[::2], axis=0)
                data_intensities = np.array(
                    [np.sum(img * self.mask) for img in images[::2]]
                )  # using list to save memory
                dark_mean = np.mean(images[1::2], axis=0)
                dark_intensities = np.array(
                    [np.sum(img * self.mask) for img in images[1::2]]
                )  # using list to save memory
                confidence = sum_1 / np.max((sum_2, 1e-10))
                if confidence < 2:
                    warnings.warn(
                        f"low confidence in distnguishing darks: {src} frac={sum_1 / sum_2}",
                        UndistinguishableWarning,
                    )
            else:
                data_mean = np.mean(images[1::2], axis=0)
                data_intensities = np.array(
                    [np.sum(img * self.mask) for img in images[1::2]]
                )  # using list to save memory
                dark_mean = np.mean(images[::2], axis=0)
                dark_intensities = np.array(
                    [np.sum(img * self.mask) for img in images[::2]]
                )  # using list to save memory
                confidence = sum_2 / np.max((sum_1, 1e-10))
                if confidence < 2:
                    warnings.warn(
                        f"low confidence in distnguishing darks: {src} frac={sum_2 / sum_1}",
                        UndistinguishableWarning,
                    )

        with h5py.File(dest, "w") as f:
            f.create_dataset(name, data=data_mean)
            f.create_dataset("dark", data=dark_mean)
            f.create_dataset(f"{name}_sum_intensities", data=data_intensities)
            f.create_dataset("dark_sum_intensities", data=dark_intensities)
            f.create_dataset("confidence", data=confidence)

    def __process_pump_probe(self, src, dest):
        with h5py.File(src, "r") as f:
            if self.discard_fist_last_img:
                images = f["entry/data/data"][()][1:-1]
            else:
                images = f["entry/data/data"][()]
            # in rare cases we observe broken images that show vertical stripes with values of 2**16-1 = 65535
            # if we find an image like that, we just drop the entire batch of images
            # specifically we check the 150th column of all the images and check how often we find the value 65535
            # if it occurs more than 3 times, we drop the file
            if np.sum((images[:, :, 150] * self.mask[:, 150]) == 65535) > 3:
                warnings.warn(f"found broken image in: {src}; skipping...")
                return

            # look at the borders of the the 10th block of 100 images and compare them
            if self.n_imgs > 1000:
                border_1 = np.sum(images[900:1000:2], axis=0)
                border_2 = np.sum(images[901:1001:2], axis=0)
            else:
                border_1 = np.sum(images[::2], axis=0)
                border_2 = np.sum(images[1::2], axis=0)
            border_1 = np.sum(border_1 * self.border_mask)
            border_2 = np.sum(border_2 * self.border_mask)
            if border_1 > border_2:
                first_image_type = "pump_on"
                confidence = border_1 / np.max((border_2, 1e-10))
                pump_on = images[::2]
                pump_off = images[1::2]
                if confidence < 100:
                    warnings.warn(
                        f"low confidence in distinguishing pump on/off: {src} frac={border_1 / border_2}",
                        UndistinguishableWarning,
                    )
            else:
                first_image_type = "pump_off"
                confidence = border_2 / np.max((border_1, 1e-10))
                pump_on = images[1::2]
                pump_off = images[::2]
                if confidence < 100:
                    warnings.warn(
                        f"low confidence in distnguishing pump on/off: {src} frac={border_2 / border_1}"
                    )
        difference_mean = np.zeros((512, 512), dtype="float")
        for on, off in zip(pump_on, pump_off):
            difference_mean += on.astype("float") - off.astype("float")
        difference_mean /= self.n_imgs
        pump_on_mean = np.mean(pump_on, axis=0)
        pump_on_intensities = np.array(
            [np.sum(img * self.mask) for img in pump_on]
        )  # using list to save memory
        pump_off_mean = np.mean(pump_off, axis=0)
        pump_off_intensities = np.array(
            [np.sum(img * self.mask) for img in pump_off]
        )  # using list to save memory

        with h5py.File(dest, "w") as f:
            f.create_dataset("pump_on", data=pump_on_mean, **hdf5plugin.Bitshuffle())
            f.create_dataset(
                "pump_on_sum_intensities",
                data=pump_on_intensities,
                **hdf5plugin.Bitshuffle(),
            )
            f.create_dataset("pump_off", data=pump_off_mean, **hdf5plugin.Bitshuffle())
            f.create_dataset(
                "pump_off_sum_intensities",
                data=pump_off_intensities,
                **hdf5plugin.Bitshuffle(),
            )
            f.create_dataset(
                "difference", data=difference_mean, **hdf5plugin.Bitshuffle()
            )
            f.create_dataset("confidence", data=confidence)
            f.create_dataset("first_image_type", data=first_image_type)


class SingleShotProcessorGen2(ThreadPoolExecutor):
    """class to process hdf5 files from single shot experiments
    no dark subtraction; no laser bg subtraction
    the whole dataset will be saved into a single hdf5 file with the following structure
    N - images per file; F - number of files to process
    /                         Group
    /confidence               Dataset {F} -> confidence that pump on/off is correctly identified
    /pump_on                  Dataset {delays, image_x, image_y} -> pump on data
    /pump_off                 Dataset {delays, image_x, image_y} -> pump off data
    /sum_ints_pump_on         Dataset {F*N/2} -> sum intensity of every pump on image
    /sum_ints_pump_off        Dataset {F*N/2} -> sum intensity of every pump off image
    /histogram_pump_on        Dataset {delays, 2^16} -> histogram of pixel intensities in pump on images
    /histogram_pump_off       Dataset {delays, 2^16} -> histogram of pixel intensities in pump off images

    Example:
    >>>from pathlib import Path
    >>>from concurrent.futures import as_completed

    >>>rundir = "/data/TiSe2_run_0010"
    >>>with SingleShotProcessorGen2([str(p) for p in Path(rundir).rglob("*ps.h5")], max_workers=1) as ssp:
    >>>    ssp.start()
    >>>    for future in as_completed(ssp.futures):
    >>>        print(ssp[future])
    """

    BORDERSIZE = 8

    def __init__(
        self,
        filelist,
        dest_file,
        mask=None,
        max_workers=None,
        discard_fist_last_img=True,
        tempfile=None,
        rois={},
    ):
        self.filelist = filelist
        if path.exists(dest_file):
            raise OSError(f"{dest_file} already exists")
        self.dest_file = dest_file
        self.discard_fist_last_img = discard_fist_last_img
        if tempfile is None:
            self.tempfile = Path(dest_file).stem + "_tmp.h5"
        else:
            self.tempfile = tempfile
        self.sample_dataset = h5py.File(filelist[0], "r")["entry/data/data"]
        if max_workers is None:
            # determine max workers from free memory and size of dataset
            free_memory = virtual_memory().available
            datasets_in_memory = int(
                free_memory
                / (
                    self.sample_dataset.dtype.itemsize
                    * np.prod(self.sample_dataset.shape)
                )
            )
            if datasets_in_memory == 0:
                warnings.warn(
                    "you might want to free up some system memory; you can't fit a whole dataset into it",
                    ResourceWarning,
                )
                datasets_in_memory = 1
            max_workers = datasets_in_memory
        self.img_size = self.sample_dataset.shape[1:]
        self.delays = np.array(
            sorted(set([self.__delay_from_fname(fname) for fname in self.filelist]))
        )
        if mask is None:
            self.mask = np.ones(self.img_size).astype(np.uint16)
        else:
            self.mask = mask.astype(np.uint16)
        self.rois = rois
        self.n_imgs = self.sample_dataset.shape[0]
        if discard_fist_last_img:
            self.n_imgs -= 2
        self.border_mask = np.ones(self.img_size).astype(np.uint16)
        self.border_mask[
            self.BORDERSIZE : -self.BORDERSIZE, self.BORDERSIZE : -self.BORDERSIZE
        ] = 0
        self.confidence = np.zeros(len(filelist))
        self.pump_on = np.zeros((len(self.delays), self.img_size[0], self.img_size[1]))
        self.pump_off = np.zeros((len(self.delays), self.img_size[0], self.img_size[1]))
        self.sum_ints_pump_on = np.zeros((int(len(filelist) * self.n_imgs / 2)))
        self.sum_ints_pump_off = np.zeros((int(len(filelist) * self.n_imgs / 2)))
        self.histogram_pump_on = np.zeros((len(self.delays), 2**16))
        self.histogram_pump_off = np.zeros((len(self.delays), 2**16))
        self.files_per_delay = np.zeros(len(self.delays))
        self.sum_ints_rois_pump_on = {}
        self.sum_ints_rois_pump_off = {}
        for key in self.rois:
            self.sum_ints_rois_pump_on[key] = np.zeros((int(len(filelist) * self.n_imgs / 2)))
            self.sum_ints_rois_pump_off[key] = np.zeros((int(len(filelist) * self.n_imgs / 2)))

        self.futures = {}
        super().__init__(max_workers=max_workers)

    def __getitem__(self, key):
        return self.futures[key]

    def watch(self):
        Thread(target=self.__watch).start()

    def __watch(self):
        for future in as_completed(self.futures):
            if future.exception():
                raise future.exception()

    def submit(self, filename):
        return super().submit(self.__worker, filename)

    def start(self):
        self.futures = {self.submit(fname): fname for fname in self.filelist}
        self.watch()

    def shutdown(self, *args, **kwargs):
        super().shutdown(*args, **kwargs)
        self.__save(self.dest_file)

    def __worker(self, filename):
        if "pumpon" in filename:
            self.__process_pump_probe(filename)
        else:
            raise NotImplementedError(f"don't know what to do with {filename}")

    def __save(self, file, overwrite=False, **kwargs):
        if overwrite:
            if path.exists(file):
                remove(file)
        with h5py.File(file, "x") as f:
            pump_on_group = f.create_group("pump_on")
            pump_off_group = f.create_group("pump_off")
            f.create_dataset(
                "confidence", data=self.confidence, **hdf5plugin.Bitshuffle()
            )
            f.create_dataset("mask", data=self.mask, **hdf5plugin.Bitshuffle())
            pump_on_group.create_dataset(
                "avg_intensities", data=self.pump_on, **hdf5plugin.Bitshuffle()
            )
            pump_on_group.create_dataset(
                "sum_intensities", data=self.sum_ints_pump_on, **hdf5plugin.Bitshuffle()
            )
            pump_on_group.create_dataset(
                "histogram", data=self.histogram_pump_on, **hdf5plugin.Bitshuffle()
            )
            pump_off_group.create_dataset(
                "avg_intensities", data=self.pump_off, **hdf5plugin.Bitshuffle()
            )
            pump_off_group.create_dataset(
                "sum_intensities",
                data=self.sum_ints_pump_off,
                **hdf5plugin.Bitshuffle(),
            )
            pump_off_group.create_dataset(
                "histogram", data=self.histogram_pump_off, **hdf5plugin.Bitshuffle()
            )
            if self.rois:
                roi_pump_on_group = pump_on_group.create_group("rois")
                roi_pump_off_group = pump_off_group.create_group("rois")
                for key, val in self.sum_ints_rois_pump_on.items():
                    roi_pump_on_group.create_dataset(key, data=val, **hdf5plugin.Bitshuffle())
                for key, val in self.sum_ints_rois_pump_off.items():
                    roi_pump_off_group.create_dataset(key, data=val, **hdf5plugin.Bitshuffle())
            for key, val in kwargs.items():
                f.create_dataset(key, data=val)

    def __tempsave(self, **kwargs):
        self.__save(self.tempfile, overwrite=True, **kwargs)

    def __process_pump_probe(self, src):
        with h5py.File(src, "r") as f:
            if self.n_imgs > 1000:
                border_1 = np.sum(f["entry/data/data"][900:1000:2], axis=0)
                border_2 = np.sum(f["entry/data/data"][901:1001:2], axis=0)
            else:
                border_1 = np.sum(f["entry/data/data"][::2], axis=0)
                border_2 = np.sum(f["entry/data/data"][1::2], axis=0)
        # look at the borders of the the 10th block of 100 images and compare them
        border_1 = np.sum(border_1 * self.border_mask)
        border_2 = np.sum(border_2 * self.border_mask)
        if border_1 > border_2:
            confidence = border_1 / np.max((border_2, 1e-10))
            if self.discard_fist_last_img:
                pump_on_slice = slice(2, -1, 2)
                pump_off_slice = slice(1, -2, 2)
            else:
                pump_on_slice = slice(0, None, 2)
                pump_off_slice = slice(1, None, 2)
        else:
            confidence = border_2 / np.max((border_1, 1e-10))
            if self.discard_fist_last_img:
                pump_on_slice = slice(1, -2, 2)
                pump_off_slice = slice(2, -1, 2)
            else:
                pump_on_slice = slice(1, None, 2)
                pump_off_slice = slice(0, None, 2)
        if confidence < 100:
            warnings.warn(
                f"low confidence in distinguishing pump on/off: {src} frac={border_1 / border_2}",
                UndistinguishableWarning,
            )
        file_index = self.filelist.index(src)
        delay_index = np.where(self.delays == self.__delay_from_fname(src))[0][0]
        sum_int_slice = slice(
            int(file_index * self.n_imgs / 2), int((file_index + 1) * self.n_imgs / 2)
        )
        self.files_per_delay[delay_index] += 1
        self.confidence[file_index] = confidence

        with h5py.File(src, "r") as f:
            pump_on_images = f["entry/data/data"][pump_on_slice]
        if self.__check_image_integrity(pump_on_images):
            norm_values = masked_sum(pump_on_images, self.mask)
            self.pump_on[delay_index] += normed_sum(pump_on_images, norm_values)
            self.sum_ints_pump_on[sum_int_slice] = norm_values
            self.histogram_pump_on[delay_index] += masked_histogram(pump_on_images, self.mask)
            for key, slices in self.rois.items():
                self.sum_ints_rois_pump_on[key][sum_int_slice] = indexed_masked_sum(pump_on_images, slices, self.mask)
        else:
            self.sum_ints_pump_on[sum_int_slice] = np.NaN
            for key in self.rois:
                self.sum_ints_rois_pump_on[key][sum_int_slice] = np.NaN
            warnings.warn(f"found broken image in {src}; skipping...")
        del pump_on_images

        with h5py.File(src, "r") as f:
            pump_off_images = f["entry/data/data"][pump_off_slice]
        if self.__check_image_integrity(pump_off_images):
            norm_values = masked_sum(pump_off_images, self.mask)
            self.pump_off[delay_index] += normed_sum(pump_off_images, norm_values)
            self.sum_ints_pump_off[sum_int_slice] = norm_values
            self.histogram_pump_off[delay_index] += masked_histogram(pump_off_images, self.mask)
            for key, slices in self.rois.items():
                self.sum_ints_rois_pump_off[key][sum_int_slice] = indexed_masked_sum(pump_off_images, slices, self.mask)
        else:
            self.sum_ints_pump_off[sum_int_slice] = np.NaN
            for key in self.rois:
                self.sum_ints_rois_pump_off[key][sum_int_slice] = np.NaN
            warnings.warn(f"found broken image in {src}; skipping...")

        self.__tempsave(progress=file_index / len(self.filelist))

    @staticmethod
    def __delay_from_fname(fname):
        return float(fname.split(r"/")[-1][7:17])

    def __check_image_integrity(self, images):
        """
        in rare cases we observe broken images that show vertical stripes with values of 2**16-1 = 65535
        if we find an image like that, we just drop the entire batch of images
        specifically we check the 150th column of all the images and check how often we find the value 65535
        if it occurs more than 3 times, we drop the file
        """
        if np.sum((images[:, :, 150] * self.mask[:, 150]) == 65535) > 3:
            return False
        return True


class SingleShotDataset:
    """class to load a single shot experiment processed with `SingleShotProcessor`"""

    log_timestamp_pattern = re.compile(r"\d*-\d*-\d* \d*:\d*:\d*")
    log_delay_pattern = re.compile(r"time-delay -?\d*.?\d*ps")
    log_scan_pattern = re.compile(r"scan \d*")

    def __init__(
        self,
        basedir,
        mask=None,
        normalize=True,
        progress=False,
        correct_dark=False,
        correct_laser=False,
        scans=slice(None),
    ):
        self.basedir = basedir

        if scans is not slice(None) and not isinstance(scans, slice):
            if isinstance(scans, list):
                pass
            elif isinstance(scans, Iterable):
                scans = slice(*scans)
            else:
                scans = slice(scans)

        scan_paths = [
            entry
            for entry in sorted(listdir(basedir))
            if entry.startswith("scan_") and path.isdir(path.join(basedir, entry))
        ][scans]
        h5_paths = []
        for p in scan_paths:
            for file in listdir(path.join(self.basedir, p)):
                if file.endswith("_processed.h5"):
                    h5_paths.append(path.join(self.basedir, p, file))
        h5_paths = sorted(h5_paths)

        # detect image shape and image count per file
        with h5py.File(h5_paths[0], "r") as f:
            self.img_shape = f["pump_on"].shape
            self.imgs_per_file = f["pump_on_sum_intensities"].shape[0]

        self.dark = self.__load_diagnostic("dark", ["laser_background", "pump_off"])
        self.laser_only = self.__load_diagnostic("laser_bg", "laser_background")
        self.pump_off_diagnostic = self.__load_diagnostic("pump_off", "pump_off")

        if mask is None:
            self.mask = np.ones(self.img_shape).astype("int")
        else:
            self.mask = mask.astype("int")

        self.delays = np.array(
            sorted(set([self.__delay_from_fname(fname) for fname in h5_paths]))
        )

        realtime_idxs = {}
        self.timestamps = []
        self.diagnostic_timestamps = []
        with open(path.join(basedir, "experiment.log")) as f:
            i = 0
            for line in f:
                if "pump on image series acquired at scan " in line:
                    scan = int(self.log_scan_pattern.findall(line)[0][5:])
                    delay = float(self.log_delay_pattern.findall(line)[0][11:-2])
                    realtime_idxs[
                        path.join(
                            self.basedir,
                            f"scan_{scan:04d}",
                            f"pumpon_{delay:+010.3f}ps_processed.h5",
                        )
                    ] = i
                    i += 1
                    timestamp = self.log_timestamp_pattern.findall(line)[0]
                    self.timestamps.append(self.__str_to_datetime(timestamp))
                if "laser background image series acquired" in line:
                    timestamp = self.log_timestamp_pattern.findall(line)[0]
                    self.diagnostic_timestamps.append(self.__str_to_datetime(timestamp))
        self.timedeltas = [ts - self.timestamps[0] for ts in self.timestamps]

        # allocate memory and load actual images
        self.diffdata = np.zeros(
            (len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float
        )
        self.pump_on = np.zeros(
            (len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float
        )
        self.pump_off = np.zeros(
            (len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float
        )

        # self.all_pump_on_imgs = np.zeros((len(h5_paths), self.img_shape[0], self.img_shape[1]), dtype=float)
        # self.all_pump_off_imgs = np.zeros((len(h5_paths), self.img_shape[0], self.img_shape[1]), dtype=float)
        self.all_diffimgs = np.zeros(
            (len(h5_paths), self.img_shape[0], self.img_shape[1]), dtype=float
        )

        self.real_time_intensities = np.zeros(len(realtime_idxs))
        self.real_time_delays = np.zeros(len(realtime_idxs))
        files_per_delay = np.zeros(len(self.delays)).squeeze()
        if progress:
            iterable = tqdm(h5_paths, desc=f"loading {self.basedir}")
        else:
            iterable = h5_paths
        for i, h5path in enumerate(iterable):
            with h5py.File(h5path, "r") as f:
                img_pump_on = f["pump_on"][()]
                img_pump_off = f["pump_off"][()]
                img_pump_on_sum_intensities = f["pump_on_sum_intensities"][()]
                img_pump_off_sum_intensities = f["pump_off_sum_intensities"][()]
                diffimg = f["difference"][()]
            if correct_dark:
                img_pump_on -= self.dark
                img_pump_off -= self.dark
            if correct_laser:
                img_pump_on -= self.laser_only
                img_pump_off -= self.laser_only
            self.real_time_intensities[realtime_idxs[h5path]] = np.sum(
                img_pump_on * self.mask
            )
            self.real_time_delays[realtime_idxs[h5path]] = self.__delay_from_fname(
                h5path
            )
            if normalize:
                img_pump_on /= np.mean(img_pump_on_sum_intensities)
                img_pump_off /= np.mean(img_pump_off_sum_intensities)
            self.pump_on[
                np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
            ] += img_pump_on
            self.pump_off[
                np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
            ] += img_pump_off
            self.diffdata[
                np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
            ] += diffimg
            # self.all_pump_on_imgs[i] = img_pump_on
            # self.all_pump_off_imgs[i] = img_pump_off
            self.all_diffimgs[i] = diffimg
            files_per_delay[
                np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
            ] += 1
        self.pump_on /= files_per_delay[:, None, None]
        self.pump_off /= files_per_delay[:, None, None]
        self.diffdata /= files_per_delay[:, None, None]
        self.mean_img = np.mean(self.pump_on, axis=0)
        self.mean_diffimg = np.mean(self.diffdata, axis=0)

    def save(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("time_points", data=self.delays)
            f.create_dataset("valid_mask", data=self.mask)
            proc_group = f.create_group("processed")
            proc_group.create_dataset(
                "equilibrium", data=np.mean(self.pump_off, axis=0)
            )
            proc_group.create_dataset(
                "intensity", data=np.moveaxis(self.pump_on, 0, -1)
            )
            realtime_group = f.create_group("real_time")
            realtime_group.create_dataset(
                "minutes", data=[td.total_seconds() / 60 for td in self.timedeltas]
            )
            realtime_group.create_dataset("intensity", data=self.real_time_intensities)
            realtime_group.create_dataset("time_points", data=self.real_time_delays)

    @staticmethod
    def __str_to_datetime(s):
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    def __load_diagnostic(self, dataset_key, directories):
        img_count = 0
        image = np.zeros(self.img_shape)
        if isinstance(directories, str):
            directories = [directories]
        for directory in directories:
            for filename in listdir(path.join(self.basedir, directory)):
                if filename.endswith("_processed.h5"):
                    with h5py.File(
                        path.join(self.basedir, directory, filename), "r"
                    ) as f:
                        image += f[dataset_key][()]
                        img_count += self.imgs_per_file
        return image / img_count

    @staticmethod
    def __delay_from_fname(fname):
        return float(fname.split(r"/")[-1][7:17])
