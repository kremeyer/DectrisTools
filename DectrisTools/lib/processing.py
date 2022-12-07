"""
module for data processing tools
"""
from os import path, mkdir
from pathlib import Path
import numpy as np
import hdf5plugin
import h5py
from .computation import masked_histogram, masked_sum, normed_sum


"""
masked_histogram(images, mask)

generates an array of occurences of pixel values in the masked images; bins are integers from 0 to (2**16)-1

Parameters
----------
images: array_like
    3d array of images; dtype hast to be np.uint16
slices: indexable of two slices
    slices that the images and mask will be indexed with
mask: array_like; dtype hast to be np.uint16
    2d array of the mask that will be applied to the images; must be the same shape as each image

Returns
-------
array_like
    1d array of the occurences of pixel values in the masked and indexed images; bins are integers from 0 to (2**16)-1
"""


"""
masked_sum(images, mask)

sum of a 3d-array along the 2 last axis; a 2d mask will be applied before summation

Parameters
----------
images: array_like
    3d array of images; dtype hast to be np.uint16
mask: array_like; dtype hast to be np.uint16
    2d array of the mask that will be applied to the images; must be the same shape as each image
    
Returns
-------
array_like
    1d array of the sums in the masked images
"""


"""
normed_sum(images, norm_values)

sum of a 3d-array along the first axis; each image will be normalized to the corresponding value before the summation

Parameters
----------
images: array_like
    3d array of images; dtype hast to be np.uint16
norm_values: array_like
    1d array values to normalize the images to; dtype has to be np.float32
    
Returns
-------
array_like
    2d array of the sum of the normalized images
"""


def indexed_masked_sum(images, slices, mask):
    """calls the masked_sum function but indexes the images and masks array before passing them

    Parameters
    ----------
    images: array_like
        3d array of images
    slices: indexable of two slices
        slices that the images and mask will be indexed with
    mask: array_like
        2d array of the mask that will be applied to the images; must be the same shape as each image

    Returns
    -------
    array_like
        1d array of the sums in the masked and indexed images
    """
    return masked_sum(images[:, slices[0], slices[1]], mask[slices[0], slices[1]])


def indexed_masked_histogram(images, slices, mask):
    """calls the masked_histogram function but indexes the images and masks array before passing them

    Parameters
    ----------
    images: array_like
        3d array of images
    slices: indexable of two slices
        slices that the images and mask will be indexed with
    mask: array_like
        2d array of the mask that will be applied to the images; must be the same shape as each image

    Returns
    -------
    array_like
        1d array of the occurences of pixel values in the masked and indexed images; bins integers from 0 to (2**16)-1
    """
    return masked_histogram(images[:, slices[0], slices[1]], mask[slices[0], slices[1]])


class AlreadyProcessedWarning(Warning):
    """warning to throw when encountering an already processed file that will not be overwritten"""


class UndistinguishableWarning(Warning):
    """warning to throw when encountering a stack of image where pump on and off cannot be distinguished with high
    confidence
    """


class BrokenImageWarning(Warning):
    """warning to throw when encountering broken images"""


def process_pump_probe(src, tempdir="tmp", mask=None, border_size=8, discard_first_last_img=True, rois=None):
    """data reduction function to process raw files recorded by the dectris camera into a file of reduced size that is
    to be collected and merged into a full dataset later

    Parameters
    ----------
    src: str
        source file path
    tempdir: str, optional
        target directory
    mask: array_like, optional
        mask to select valid parts of the diffraction images; 2d array with dtype np.uint16
    border_size: int, optional
        width of the mask that is used to distinguish pump on and off images
    discard_first_last_img: bool, optional
        in practice the first and last images of a file tend to be completely dark, thus they can be ignored with this
        option
    rois: dict, optional
        dictionary with 2 tuples of slices with ROIs in the images that will be analyzed

    Returns
    -------
    list
        list of warnings generated during processing
    """
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
                f"low confidence in distinguishing pump on/off:\
                {src} frac={max(border_1 / border_2, border_2 / border_1)}"
            )
        )

    sum_ints_rois_pump_on = {}
    sum_ints_rois_pump_off = {}
    histogram_rois_pump_on = {}
    histogram_rois_pump_off = {}

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
            histogram_rois_pump_on[key] = indexed_masked_histogram(pump_on_images, slices, mask)
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
            histogram_rois_pump_off[key] = indexed_masked_histogram(pump_off_images, slices, mask)
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
            for key, slices in rois.items():
                key_group = roi_pump_on_group.create_group(key)
                key_group.create_dataset(
                    "sum_intensities",
                    data=sum_ints_rois_pump_on[key],
                    **hdf5plugin.Bitshuffle(),
                )
                key_group.create_dataset(
                    "slices",
                    data=(slice_to_tuple(slices[0]), slice_to_tuple(slices[1])),
                    **hdf5plugin.Bitshuffle(),
                )
                key_group.create_dataset(
                    "histogram",
                    data=histogram_rois_pump_on[key],
                    **hdf5plugin.Bitshuffle(),
                )
                key_group = roi_pump_off_group.create_group(key)
                key_group.create_dataset(
                    "sum_intensities",
                    data=sum_ints_rois_pump_off[key],
                    **hdf5plugin.Bitshuffle(),
                )
                key_group.create_dataset(
                    "slices",
                    data=(slice_to_tuple(slices[0]), slice_to_tuple(slices[1])),
                    **hdf5plugin.Bitshuffle(),
                )
                key_group.create_dataset(
                    "histogram",
                    data=histogram_rois_pump_off[key],
                    **hdf5plugin.Bitshuffle(),
                )
        return warns


def collect_results(tempdir, result_file):
    """collect all files generated by `process_pump_probe` in a given directory and merge them into a single file

    Parameters
    ----------
    tempdir: str
        path to directory with generated files
    result_file: str
        output file
    """
    processed_files = sorted(list(Path(tempdir).rglob("**/*.h5")))

    delays = np.array(sorted(set([delay_from_fname(str(fname)) for fname in processed_files])))

    # get general info from first temp file
    rois = {}
    with h5py.File(processed_files[-1], "r") as f:
        img_size = f["mask"].shape
        mask = f["mask"][()]
        n_imgs = f["pump_on/sum_intensities"].shape[0]
        if "rois" in f["pump_on"]:
            for roi in f["pump_on/rois"]:
                slice_info = f[f"pump_on/rois/{roi}/slices"][()]
                rois[roi] = (
                    tuple_to_slice(slice_info[0]),
                    tuple_to_slice(slice_info[1]),
                )

    # allocate memory for final data
    confidence = np.zeros(len(processed_files))
    pump_on = np.zeros((len(delays), img_size[0], img_size[1]))
    pump_off = np.zeros((len(delays), img_size[0], img_size[1]))
    sum_ints_pump_on = np.zeros((int(len(processed_files) * n_imgs)))
    sum_ints_pump_off = np.zeros((int(len(processed_files) * n_imgs)))
    histogram_pump_on = np.zeros((len(delays), 2**16))
    histogram_pump_off = np.zeros((len(delays), 2**16))
    files_per_delay = np.zeros(len(delays))
    sum_ints_rois_pump_on = {}
    sum_ints_rois_pump_off = {}
    histograms_rois_pump_on = {}
    histograms_rois_pump_off = {}
    for key in rois:
        sum_ints_rois_pump_on[key] = np.zeros((int(len(processed_files) * n_imgs)))
        sum_ints_rois_pump_off[key] = np.zeros((int(len(processed_files) * n_imgs)))
        histograms_rois_pump_on[key] = np.zeros((len(delays), 2**16))
        histograms_rois_pump_off[key] = np.zeros((len(delays), 2**16))

    # read temporary files
    for file in processed_files:
        try:
            file_index = processed_files.index(file)
            delay_index = np.where(delays == delay_from_fname(str(file)))[0][0]
            sum_int_slice = slice(int(file_index * n_imgs), int((file_index + 1) * n_imgs))
            with h5py.File(path.join(tempdir, file), "r") as f:
                confidence[file_index] = f["confidence"][()]
                pump_on[delay_index] += f["pump_on/avg_intensities"][()]
                sum_ints_pump_on[sum_int_slice] = f["pump_on/sum_intensities"][()]
                histogram_pump_on[delay_index] += f["pump_on/histogram"][()]
                for key in rois:
                    sum_ints_rois_pump_on[key][sum_int_slice] = f[f"pump_on/rois/{key}/sum_intensities"][()]
                    histograms_rois_pump_on[key][delay_index] += f[f"pump_on/rois/{key}/histogram"][()]
                pump_off[delay_index] += f["pump_off/avg_intensities"][()]
                sum_ints_pump_off[sum_int_slice] = f["pump_off/sum_intensities"][()]
                histogram_pump_off[delay_index] += f["pump_off/histogram"][()]
                for key in rois:
                    sum_ints_rois_pump_off[key][sum_int_slice] = f[f"pump_off/rois/{key}/sum_intensities"][()]
                    histograms_rois_pump_off[key][delay_index] += f[f"pump_on/rois/{key}/histogram"][()]
            files_per_delay[delay_index] += 1
        except (BlockingIOError, OSError):
            continue
    pump_on /= files_per_delay[:, None, None]
    pump_off /= files_per_delay[:, None, None]

    # write final output file
    with h5py.File(result_file, "x") as f:
        f.create_dataset("confidence", data=confidence, **hdf5plugin.Bitshuffle())
        f.create_dataset("mask", data=mask, **hdf5plugin.Bitshuffle())
        f.create_dataset("delays", data=delays, **hdf5plugin.Bitshuffle())
        pump_on_group = f.create_group("pump_on")
        pump_off_group = f.create_group("pump_off")
        rois_on_group = pump_on_group.create_group("rois")
        rois_off_group = pump_off_group.create_group("rois")
        pump_on_group.create_dataset("avg_intensities", data=pump_on, **hdf5plugin.Bitshuffle())
        pump_on_group.create_dataset("sum_intensities", data=sum_ints_pump_on, **hdf5plugin.Bitshuffle())
        pump_on_group.create_dataset("histogram", data=histogram_pump_on, **hdf5plugin.Bitshuffle())
        for key, slices in rois.items():
            key_group = rois_on_group.create_group(key)
            key_group.create_dataset(
                "sum_intensities",
                data=sum_ints_rois_pump_on[key],
                **hdf5plugin.Bitshuffle(),
            )
            key_group.create_dataset(
                "histogram",
                data=histograms_rois_pump_on[key],
                **hdf5plugin.Bitshuffle(),
            )
            key_group.create_dataset("slices", data=(slice_to_tuple(slices[0]), slice_to_tuple(slices[1])))
        pump_off_group.create_dataset("avg_intensities", data=pump_off, **hdf5plugin.Bitshuffle())
        pump_off_group.create_dataset("sum_intensities", data=sum_ints_pump_off, **hdf5plugin.Bitshuffle())
        pump_off_group.create_dataset("histogram", data=histogram_pump_off, **hdf5plugin.Bitshuffle())
        for key, slices in rois.items():
            key_group = rois_off_group.create_group(key)
            key_group.create_dataset(
                "sum_intensities",
                data=sum_ints_rois_pump_off[key],
                **hdf5plugin.Bitshuffle(),
            )
            key_group.create_dataset(
                "histogram",
                data=histograms_rois_pump_off[key],
                **hdf5plugin.Bitshuffle(),
            )
            key_group.create_dataset("slices", data=(slice_to_tuple(slices[0]), slice_to_tuple(slices[1])))


def check_image_integrity(images, mask):
    """
    checks a stack of images for corruption

    in rare cases we observe broken images that show vertical stripes with values of 2**16-1 = 65535
    if we find an image like that, we just drop the entire batch of images
    specifically we check the 150th column of all the images and check how often we find the value 65535
    if it occurs more than 3 times, we drop the file

    Parameters
    ----------
    images: array_like
        3d array of images
    mask: array_like
        2d array of the mask that will be applied to the images; must be the same shape as each image

    Returns
    -------
    bool
        False if no corrupted image was found; True otherwise
    """
    if np.sum((images[:, :, 150] * mask[:, 150]) == 65535) > 3:
        return False
    return True


def delay_from_fname(fname):
    """parses a filename and returns the corresponding delay

    Parameters
    ----------
    fname: str
        path to the filename to be parsed

    Returns
    -------
    float
        time delay
    """
    return float(fname.split(r"/")[-1][7:17])


def slice_to_tuple(sl):
    """converts an instance of a slice to a tuple; Nones will be replaced by NaNs

    Parameters
    ----------
    sl: slice
        slice to convert

    Returns
    -------
    tuple
        3-tuple of (start, stop, step)
    """
    if sl.start is None:
        start = np.NaN
    else:
        start = int(sl.start)
    if sl.stop is None:
        stop = np.NaN
    else:
        stop = int(sl.stop)
    if sl.step is None:
        step = np.NaN
    else:
        step = int(sl.step)
    return start, stop, step


def tuple_to_slice(tup):
    """converts a tuple to a slice; NaNs will be replaced by Nones

    Parameters
    ----------
    tup: tuple
        3-tuple with start, stop and step

    Returns
    -------
    slice
        converted slice
    """
    if np.isnan(tup[0]):
        start = None
    else:
        start = tup[0]
    if np.isnan(tup[1]):
        stop = None
    else:
        stop = tup[1]
    if np.isnan(tup[2]):
        step = None
    else:
        step = tup[2]
    return slice(start, stop, step)


def generate_bordermask(mask_shape, border_size):
    """generate a 2d-array of zeros with a border of ones

    Parameters
    ----------
    mask_shape: tuple
        2-tuple corresponding to the output mask shape
    border_size: int
        width of the border of ones

    Returns
    -------
    np.array
        bordermask
    """
    bordermask = np.ones(mask_shape).astype(np.uint16)
    bordermask[border_size:-border_size, border_size:-border_size] = 0
    return bordermask


# TODO: introduce function to read files from logfile, so they are in ordered in lab time

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
