"""
module for data processing tools
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import warnings
from os import path, listdir
from pathlib import Path
import re
from datetime import datetime
from psutil import virtual_memory
import numpy as np
import hdf5plugin
import h5py
from tqdm import tqdm


class AlreadyProcessedWarning(Warning):
    pass


class UndistinguishableWarning(Warning):
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

    BORDERSIZE = 5

    def __init__(self, filelist, mask=None, max_workers=None, ignore_existing=False):
        self.filelist = filelist
        self.ignore_existing = ignore_existing
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
                    "you might want to free up some system memory; you can fit a whole dataset into it",
                    ResourceWarning,
                )
                datasets_in_memory = 1
            max_workers = datasets_in_memory
        self.img_size = self.sample_dataset.shape[1:]
        if mask is None:
            self.mask = np.ones(self.img_size)
        else:
            self.mask = mask
        self.n_imgs = self.sample_dataset.shape[0]
        self.border_mask = np.ones(self.img_size).astype("bool")
        self.border_mask[
            self.BORDERSIZE: -self.BORDERSIZE, self.BORDERSIZE: -self.BORDERSIZE
        ] = False
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
        processed_filename = (
            f'{path.join(dirname, Path(filename).stem)}_processed.h5'
        )
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
            images = f["entry/data/data"][()]
            # look at the intensity sum in the first images and compare them; darks will be dark...
            if self.n_imgs > 100:
                sum_1 = np.sum(images[:100:2])
                sum_2 = np.sum(images[1:101:2])
            else:
                sum_1 = np.sum(images[::2])
                sum_2 = np.sum(images[1::2])
            if sum_1 > sum_2:
                data_mean = np.mean(images[::2], axis=0)
                data_intensities = np.array([np.sum(img*self.mask) for img in images[::2]])  # using list to save memory
                dark_mean = np.mean(images[1::2], axis=0)
                dark_intensities = np.array([np.sum(img*self.mask) for img in images[1::2]])  # using list to save memory
                if sum_1 / np.max((sum_2, 1e-10)) < 100:
                    warnings.warn(
                        "low confidence in distnguishing pump on/off data",
                        UndistinguishableWarning,
                    )
            else:
                data_mean = np.mean(images[1::2], axis=0)
                data_intensities = np.array([np.sum(img*self.mask) for img in images[1::2]])  # using list to save memory
                dark_mean = np.mean(images[::2], axis=0)
                dark_intensities = np.array([np.sum(img*self.mask) for img in images[::2]])  # using list to save memory
                if sum_2 / np.max((sum_2, 1e-10)) < 100:
                    warnings.warn(
                        "low confidence in distnguishing pump on/off data",
                        UndistinguishableWarning,
                    )

        with h5py.File(dest, "w") as f:
            f.create_dataset(name, data=data_mean)
            f.create_dataset('dark', data=dark_mean)
            f.create_dataset(f'{name}_sum_intensities', data=data_intensities)
            f.create_dataset("dark_sum_intensities", data=dark_intensities)

    def __process_pump_probe(self, src, dest):
        with h5py.File(src, "r") as f:
            images = f["entry/data/data"][()]
            # look at the borders of the first 100 images and compare them
            if self.n_imgs > 100:
                border_1 = np.sum(images[:100:2], axis=0)
                border_2 = np.sum(images[1:101:2], axis=0)
            else:
                border_1 = np.sum(images[::2], axis=0)
                border_2 = np.sum(images[1::2], axis=0)
            border_1 = np.sum(border_1[self.border_mask])
            border_2 = np.sum(border_2[self.border_mask])
            if border_1 > border_2:
                confidence = border_1 / np.max((border_2, 1e-10))
                pump_on = images[::2]
                pump_off = images[1::2]
                if confidence < 100:
                    warnings.warn(
                        f"low confidence in distnguishing pump on/off: {src} frac={border_1 / border_2}",
                        UndistinguishableWarning,
                    )
            else:
                confidence = border_2 / np.max((border_1, 1e-10))
                pump_on = images[1::2]
                pump_off = images[::2]
                if confidence < 100:
                    warnings.warn(f"low confidence in distnguishing pump on/off: {src} frac={border_2 / border_1}")
        difference_mean = np.zeros((512, 512), dtype='float')
        for on, off in zip(pump_on, pump_off):
            difference_mean += (on.astype('float') - off.astype('float'))
        difference_mean /= self.n_imgs
        pump_on_mean = np.mean(pump_on, axis=0)
        pump_on_intensities = np.array([np.sum(img*self.mask) for img in pump_on])  # using list to save memory
        pump_off_mean = np.mean(pump_off, axis=0)
        pump_off_intensities = np.array([np.sum(img*self.mask) for img in pump_off])  # using list to save memory

        with h5py.File(dest, "w") as f:
            f.create_dataset("pump_on", data=pump_on_mean, **hdf5plugin.Bitshuffle())
            f.create_dataset("pump_on_sum_intensities", data=pump_on_intensities, **hdf5plugin.Bitshuffle())
            f.create_dataset("pump_off", data=pump_off_mean, **hdf5plugin.Bitshuffle())
            f.create_dataset("pump_off_sum_intensities", data=pump_off_intensities, **hdf5plugin.Bitshuffle())
            f.create_dataset(
                "difference", data=difference_mean, **hdf5plugin.Bitshuffle()
            )
            f.create_dataset("confidence", data=confidence)


class SingleShotDataset:
    """class to load a single shot experiment processed with `SingleShotProcessor`
    """
    log_timestamp_pattern = re.compile(r"\d*-\d*-\d* \d*:\d*:\d*")
    log_delay_pattern = re.compile(r"time-delay -?\d*.?\d*ps")
    log_scan_pattern = re.compile(r"scan \d*")

    def __init__(self, basedir, mask=None, normalize=True, progress=False, correct_dark=True, correct_laser=True):
        self.basedir = basedir

        h5_paths = []
        for entry in listdir(self.basedir):
            if entry.startswith("scan_"):
                for file in listdir(path.join(self.basedir, entry)):
                    if file.endswith('_processed.h5'):
                        h5_paths.append(path.join(self.basedir, entry, file))
        h5_paths = sorted(h5_paths)

        # detect image shape and image count per file
        with h5py.File(h5_paths[0], 'r') as f:
            self.img_shape = f['pump_on'].shape
            self.imgs_per_file = f['pump_on_sum_intensities'].shape[0]

        self.dark = self.__load_diagnostic('dark', ['laser_background', 'pump_off'])
        self.laser_only = self.__load_diagnostic('laser_bg', 'laser_background')
        self.pump_off_diagnostic = self.__load_diagnostic('pump_off', 'pump_off')
        self.all_pump_on_imgs = []
        self.all_pump_off_imgs = []
        self.all_diffimgs = []

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
                    delay = float(
                        self.log_delay_pattern.findall(line)[0][11:-2]
                    )
                    realtime_idxs[
                        path.join(
                            self.basedir,
                            f"scan_{scan:04d}",
                            f"pumpon_{delay:+010.3f}ps_processed.h5",
                        )
                    ] = i
                    i += 1
                    timestamp = self.log_timestamp_pattern.findall(line)[0]
                    self.timestamps.append(
                        self.__str_to_datetime(timestamp)
                    )
                if "laser background image series acquired" in line:
                    timestamp = self.log_timestamp_pattern.findall(line)[0]
                    self.diagnostic_timestamps.append(
                        self.__str_to_datetime(timestamp)
                    )
        self.timedeltas = [ts-self.timestamps[0] for ts in self.timestamps]

        # allocate memory and load actual images
        self.diffdata = np.zeros((len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float)
        self.pump_on = np.zeros((len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float)
        self.pump_off = np.zeros((len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float)
        self.real_time_intensities = np.zeros(len(realtime_idxs))
        self.real_time_delays = np.zeros(len(realtime_idxs))
        files_per_delay = np.zeros(len(self.delays)).squeeze()
        if progress:
            iterable = tqdm(h5_paths, desc=f'loading {self.basedir}')
        else:
            iterable = h5_paths
        for h5path in iterable:
            with h5py.File(h5path, 'r') as f:
                img_pump_on = f['pump_on'][()]
                img_pump_off = f['pump_off'][()]
                img_pump_on_sum_intensities = f['pump_on_sum_intensities'][()]
                img_pump_off_sum_intensities = f['pump_off_sum_intensities'][()]
                diffimg = f['difference'][()]
            if correct_dark:
                img_pump_on -= self.dark
                img_pump_off -= self.dark
                # diffimg -= self.dark
            if correct_laser:
                img_pump_on -= self.laser_only
                img_pump_off -= self.laser_only
                # diffimg -= self.laser_only
            self.real_time_intensities[realtime_idxs[h5path]] = np.sum(img_pump_on*self.mask)
            self.real_time_delays[realtime_idxs[h5path]] = self.__delay_from_fname(h5path)
            if normalize:
                img_pump_on /= np.mean(img_pump_on_sum_intensities)
                img_pump_off /= np.mean(img_pump_off_sum_intensities)
            self.pump_on[np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]] += img_pump_on
            self.pump_off[np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]] += img_pump_off
            self.diffdata[np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]] += diffimg
            self.all_pump_on_imgs.append(img_pump_on)
            self.all_pump_off_imgs.append(img_pump_off)
            self.all_diffimgs.append(diffimg)
            files_per_delay[np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]] += 1
        self.pump_on /= files_per_delay[:, None, None]
        self.pump_off /= files_per_delay[:, None, None]
        self.diffdata /= files_per_delay[:, None, None]
        self.mean_img = np.mean(self.pump_on, axis=0)
        self.mean_diffimg = np.mean(self.diffdata, axis=0)

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
                if filename.endswith('_processed.h5'):
                    with h5py.File(path.join(self.basedir, directory, filename), 'r') as f:
                        image += f[dataset_key][()]
                        img_count += self.imgs_per_file
        return image/img_count

    @staticmethod
    def __delay_from_fname(fname):
        return float(fname.split(r"/")[-1][7:17])
