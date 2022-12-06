"""
module for data processing tools
"""
from os import path, mkdir
from pathlib import Path
import numpy as np
import hdf5plugin
import h5py
from .computation import masked_histogram, masked_sum, normed_sum


def indexed_masked_sum(images, slices, mask):
    return masked_sum(images[:, slices[0], slices[1]], mask[slices[0], slices[1]])


class AlreadyProcessedWarning(Warning):
    pass


class UndistinguishableWarning(Warning):
    pass


class BrokenImageWarning(Warning):
    pass


def process_pump_probe(src, tempdir="tmp", mask=None, border_size=8, discard_first_last_img=True, rois=None):
    warns = []

    tempfile = path.join(tempdir, path.basename(Path(src).parent), path.basename(src))
    if not path.isdir(Path(tempfile).parent):
        try:
            # this can trigger an exception if two mpi processes create the dir at the same time
            mkdir(Path(tempfile).parent)
        except FileExistsError:
            pass
    if path.exists(tempfile):
        warns.append(AlreadyProcessedWarning(f"{src} already processed; skipping"))
        return warns

    with h5py.File(src, "r") as f:
        img_size = f["entry/data/data"].shape[1:]
        start_idx = int(f["entry/data/data"].shape[0] / 10)
        stop_idx = start_idx + 100
        border_1 = np.sum(f["entry/data/data"][start_idx:stop_idx:2], axis=0)
        border_2 = np.sum(f["entry/data/data"][start_idx + 1 : stop_idx + 1 : 2], axis=0)

    border_mask = generate_bordermask(mask.shape, border_size)
    border_1 = np.sum(border_1 * border_mask)
    border_2 = np.sum(border_2 * border_mask)
    if border_1 > border_2:
        confidence = border_1 / np.max((border_2, 1e-10))
        if discard_first_last_img:
            pump_on_slice = slice(2, -1, 2)
            pump_off_slice = slice(1, -2, 2)
        else:
            pump_on_slice = slice(0, None, 2)
            pump_off_slice = slice(1, None, 2)
    else:
        confidence = border_2 / np.max((border_1, 1e-10))
        if discard_first_last_img:
            pump_on_slice = slice(1, -2, 2)
            pump_off_slice = slice(2, -1, 2)
        else:
            pump_on_slice = slice(1, None, 2)
            pump_off_slice = slice(0, None, 2)
    if confidence < 50:
        warns.append(
            UndistinguishableWarning(
                f"low confidence in distinguishing pump on/off: {src} frac={max(border_1 / border_2, border_2 / border_1)}"
            )
        )

    sum_ints_rois_pump_on = {}
    sum_ints_rois_pump_off = {}

    if mask is None:
        mask = np.ones(img_size, dtype=np.uint16)

    with h5py.File(src, "r") as f:
        pump_on_images = f["entry/data/data"][pump_on_slice]
    if check_image_integrity(pump_on_images, mask):
        norm_values = masked_sum(pump_on_images, mask).astype(np.float32)
        pump_on = normed_sum(pump_on_images, norm_values)
        sum_ints_pump_on = norm_values
        histogram_pump_on = masked_histogram(pump_on_images, mask)
        for key, slices in rois.items():
            sum_ints_rois_pump_on[key] = indexed_masked_sum(pump_on_images, slices, mask)
    else:
        warns.append(BrokenImageWarning(f"found broken image in {src}; skipping"))
        return warns
    del pump_on_images

    with h5py.File(src, "r") as f:
        pump_off_images = f["entry/data/data"][pump_off_slice]
    if check_image_integrity(pump_off_images, mask):
        norm_values = masked_sum(pump_off_images, mask).astype(np.float32)
        pump_off = normed_sum(pump_off_images, norm_values)
        sum_ints_pump_off = norm_values
        histogram_pump_off = masked_histogram(pump_off_images, mask)
        for key, slices in rois.items():
            sum_ints_rois_pump_off[key] = indexed_masked_sum(pump_off_images, slices, mask)
    else:
        warns.append(BrokenImageWarning(f"found broken image in {src}; skipping"))
        return warns

    with h5py.File(tempfile, "x") as f:
        pump_on_group = f.create_group("pump_on")
        pump_off_group = f.create_group("pump_off")
        f.create_dataset("confidence", data=confidence)
        f.create_dataset("mask", data=mask, **hdf5plugin.Bitshuffle())
        f.create_dataset("delay", data=delay_from_fname(src))
        pump_on_group.create_dataset("avg_intensities", data=pump_on, **hdf5plugin.Bitshuffle())
        pump_on_group.create_dataset("sum_intensities", data=sum_ints_pump_on, **hdf5plugin.Bitshuffle())
        pump_on_group.create_dataset("histogram", data=histogram_pump_on, **hdf5plugin.Bitshuffle())
        pump_off_group.create_dataset("avg_intensities", data=pump_off, **hdf5plugin.Bitshuffle())
        pump_off_group.create_dataset(
            "sum_intensities",
            data=sum_ints_pump_off,
            **hdf5plugin.Bitshuffle(),
        )
        pump_off_group.create_dataset("histogram", data=histogram_pump_off, **hdf5plugin.Bitshuffle())
        if rois:
            roi_pump_on_group = pump_on_group.create_group("rois")
            roi_pump_off_group = pump_off_group.create_group("rois")
            for key, val in sum_ints_rois_pump_on.items():
                roi_pump_on_group.create_dataset(key, data=val, **hdf5plugin.Bitshuffle())
            for key, val in sum_ints_rois_pump_off.items():
                roi_pump_off_group.create_dataset(key, data=val, **hdf5plugin.Bitshuffle())
        return warns


def check_image_integrity(images, mask):
    """
    in rare cases we observe broken images that show vertical stripes with values of 2**16-1 = 65535
    if we find an image like that, we just drop the entire batch of images
    specifically we check the 150th column of all the images and check how often we find the value 65535
    if it occurs more than 3 times, we drop the file
    """
    if np.sum((images[:, :, 150] * mask[:, 150]) == 65535) > 3:
        return False
    return True


def delay_from_fname(fname):
    return float(fname.split(r"/")[-1][7:17])


def slice_to_tuple(sl):
    if sl.start is None:
        start = np.NaN
    else:
        start = int(sl.start)
    if sl.stop is None:
        stop = np.NaN
    else:
        stop = int(sl.start)
    if sl.step is None:
        step = np.NaN
    else:
        step = int(sl.start)
    return start, stop, step


def generate_bordermask(mask_shape, border_size):
    border_mask = np.ones(mask_shape).astype(np.uint16)
    border_mask[border_size:-border_size, border_size:-border_size] = 0
    return border_mask


# TODO: NEEDS MAJOR CHANGES TO ACCOMPANY THE NEW FORMAT
# class SingleShotDataset:
#     """class to load a single shot experiment processed with `SingleShotProcessor`"""
#
#     log_timestamp_pattern = re.compile(r"\d*-\d*-\d* \d*:\d*:\d*")
#     log_delay_pattern = re.compile(r"time-delay -?\d*.?\d*ps")
#     log_scan_pattern = re.compile(r"scan \d*")
#
#     def __init__(
#         self,
#         basedir,
#         mask=None,
#         normalize=True,
#         progress=False,
#         correct_dark=False,
#         correct_laser=False,
#         scans=slice(None),
#     ):
#         self.basedir = basedir
#
#         if scans is not slice(None) and not isinstance(scans, slice):
#             if isinstance(scans, list):
#                 pass
#             elif isinstance(scans, Iterable):
#                 scans = slice(*scans)
#             else:
#                 scans = slice(scans)
#
#         scan_paths = [
#             entry
#             for entry in sorted(listdir(basedir))
#             if entry.startswith("scan_") and path.isdir(path.join(basedir, entry))
#         ][scans]
#         h5_paths = []
#         for p in scan_paths:
#             for file in listdir(path.join(self.basedir, p)):
#                 if file.endswith("_processed.h5"):
#                     h5_paths.append(path.join(self.basedir, p, file))
#         h5_paths = sorted(h5_paths)
#
#         # detect image shape and image count per file
#         with h5py.File(h5_paths[0], "r") as f:
#             self.img_shape = f["pump_on"].shape
#             self.imgs_per_file = f["pump_on_sum_intensities"].shape[0]
#
#         self.dark = self.__load_diagnostic("dark", ["laser_background", "pump_off"])
#         self.laser_only = self.__load_diagnostic("laser_bg", "laser_background")
#         self.pump_off_diagnostic = self.__load_diagnostic("pump_off", "pump_off")
#
#         if mask is None:
#             self.mask = np.ones(self.img_shape).astype("int")
#         else:
#             self.mask = mask.astype("int")
#
#         self.delays = np.array(
#             sorted(set([self.__delay_from_fname(fname) for fname in h5_paths]))
#         )
#
#         realtime_idxs = {}
#         self.timestamps = []
#         self.diagnostic_timestamps = []
#         with open(path.join(basedir, "experiment.log")) as f:
#             i = 0
#             for line in f:
#                 if "pump on image series acquired at scan " in line:
#                     scan = int(self.log_scan_pattern.findall(line)[0][5:])
#                     delay = float(self.log_delay_pattern.findall(line)[0][11:-2])
#                     realtime_idxs[
#                         path.join(
#                             self.basedir,
#                             f"scan_{scan:04d}",
#                             f"pumpon_{delay:+010.3f}ps_processed.h5",
#                         )
#                     ] = i
#                     i += 1
#                     timestamp = self.log_timestamp_pattern.findall(line)[0]
#                     self.timestamps.append(self.__str_to_datetime(timestamp))
#                 if "laser background image series acquired" in line:
#                     timestamp = self.log_timestamp_pattern.findall(line)[0]
#                     self.diagnostic_timestamps.append(self.__str_to_datetime(timestamp))
#         self.timedeltas = [ts - self.timestamps[0] for ts in self.timestamps]
#
#         # allocate memory and load actual images
#         self.diffdata = np.zeros(
#             (len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float
#         )
#         self.pump_on = np.zeros(
#             (len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float
#         )
#         self.pump_off = np.zeros(
#             (len(self.delays), self.img_shape[0], self.img_shape[1]), dtype=float
#         )
#
#         # self.all_pump_on_imgs = np.zeros((len(h5_paths), self.img_shape[0], self.img_shape[1]), dtype=float)
#         # self.all_pump_off_imgs = np.zeros((len(h5_paths), self.img_shape[0], self.img_shape[1]), dtype=float)
#         self.all_diffimgs = np.zeros(
#             (len(h5_paths), self.img_shape[0], self.img_shape[1]), dtype=float
#         )
#
#         self.real_time_intensities = np.zeros(len(realtime_idxs))
#         self.real_time_delays = np.zeros(len(realtime_idxs))
#         files_per_delay = np.zeros(len(self.delays)).squeeze()
#         if progress:
#             iterable = tqdm(h5_paths, desc=f"loading {self.basedir}")
#         else:
#             iterable = h5_paths
#         for i, h5path in enumerate(iterable):
#             with h5py.File(h5path, "r") as f:
#                 img_pump_on = f["pump_on"][()]
#                 img_pump_off = f["pump_off"][()]
#                 img_pump_on_sum_intensities = f["pump_on_sum_intensities"][()]
#                 img_pump_off_sum_intensities = f["pump_off_sum_intensities"][()]
#                 diffimg = f["difference"][()]
#             if correct_dark:
#                 img_pump_on -= self.dark
#                 img_pump_off -= self.dark
#             if correct_laser:
#                 img_pump_on -= self.laser_only
#                 img_pump_off -= self.laser_only
#             self.real_time_intensities[realtime_idxs[h5path]] = np.sum(
#                 img_pump_on * self.mask
#             )
#             self.real_time_delays[realtime_idxs[h5path]] = self.__delay_from_fname(
#                 h5path
#             )
#             if normalize:
#                 img_pump_on /= np.mean(img_pump_on_sum_intensities)
#                 img_pump_off /= np.mean(img_pump_off_sum_intensities)
#             self.pump_on[
#                 np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
#             ] += img_pump_on
#             self.pump_off[
#                 np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
#             ] += img_pump_off
#             self.diffdata[
#                 np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
#             ] += diffimg
#             # self.all_pump_on_imgs[i] = img_pump_on
#             # self.all_pump_off_imgs[i] = img_pump_off
#             self.all_diffimgs[i] = diffimg
#             files_per_delay[
#                 np.argwhere(self.delays == self.__delay_from_fname(h5path))[0][0]
#             ] += 1
#         self.pump_on /= files_per_delay[:, None, None]
#         self.pump_off /= files_per_delay[:, None, None]
#         self.diffdata /= files_per_delay[:, None, None]
#         self.mean_img = np.mean(self.pump_on, axis=0)
#         self.mean_diffimg = np.mean(self.diffdata, axis=0)
#
#     def save(self, filename):
#         with h5py.File(filename, "w") as f:
#             f.create_dataset("time_points", data=self.delays)
#             f.create_dataset("valid_mask", data=self.mask)
#             proc_group = f.create_group("processed")
#             proc_group.create_dataset(
#                 "equilibrium", data=np.mean(self.pump_off, axis=0)
#             )
#             proc_group.create_dataset(
#                 "intensity", data=np.moveaxis(self.pump_on, 0, -1)
#             )
#             realtime_group = f.create_group("real_time")
#             realtime_group.create_dataset(
#                 "minutes", data=[td.total_seconds() / 60 for td in self.timedeltas]
#             )
#             realtime_group.create_dataset("intensity", data=self.real_time_intensities)
#             realtime_group.create_dataset("time_points", data=self.real_time_delays)
#
#     @staticmethod
#     def __str_to_datetime(s):
#         return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
#
#     def __load_diagnostic(self, dataset_key, directories):
#         img_count = 0
#         image = np.zeros(self.img_shape)
#         if isinstance(directories, str):
#             directories = [directories]
#         for directory in directories:
#             for filename in listdir(path.join(self.basedir, directory)):
#                 if filename.endswith("_processed.h5"):
#                     with h5py.File(
#                         path.join(self.basedir, directory, filename), "r"
#                     ) as f:
#                         image += f[dataset_key][()]
#                         img_count += self.imgs_per_file
#         return image / img_count
#
#     @staticmethod
#     def __delay_from_fname(fname):
#         return float(fname.split(r"/")[-1][7:17])
