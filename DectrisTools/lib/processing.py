"""
module for data processing tools
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import warnings
from os import path
from psutil import virtual_memory
import numpy as np
import hdf5plugin
import h5py


class AlreadyProcessedWarning(Warning):
    pass


class UndistinguishableWarning(Warning):
    pass


class SingleShotProcessor(ThreadPoolExecutor):
    """class to process hdf5 files from single shot experiments
    each hdf5 file will contain a series of images that are contain pump on/off data
    we use the pump on laser reflections at the detector border to determine which images are which and subsequently
    save three images in a new hdf5 file called {old_filename}_processed.h5:
    - a mean pump on image
    - a mean pump off image
    - a mean pump_on-pump_off image

    Example:
    from pathlib import Path
    from concurrent.futures import as_completed

    rundir = "/data/TiSe2_run_0010"
    with SingleShotProcessor([str(p) for p in Path(rundir).rglob("*.h5")], max_workers=1) as ssp:
        ssp.start()
        for future in as_completed(ssp.futures):
            print(ssp[future])
    """

    BORDERSIZE = 5

    def __init__(self, filelist, mask=None, max_workers=None):
        self.filelist = filelist
        self.sample_dataset = h5py.File(filelist[0], "r")["entry/data/data"]
        if max_workers is None:
            # determine max workers from free memory and size of dataset
            free_memory = virtual_memory().available
            with h5py.File(filelist[0], "r") as f:
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
            f'{path.join(dirname, filename.split(".")[0])}_processed.h5'
        )
        if path.exists(processed_filename):
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
                data_intensities = np.sum(images[::2], axis=(1, 2))
                dark_mean = np.mean(images[1::2], axis=0)
                dark_intensities = np.sum(images[1::2], axis=(1, 2))
                if sum_1 / sum_2 < 100:
                    warnings.warn(
                        "low confidence in distnguishing pump on/off data",
                        UndistinguishableWarning,
                    )
            else:
                data_mean = np.mean(images[1::2], axis=0)
                data_intensities = np.sum(images[1::2]*self.mask, axis=(1, 2))
                dark_mean = np.mean(images[::2], axis=0)
                dark_intensities = np.sum(images[::2]*self.mask, axis=(1, 2))
                if sum_2 / sum_1 < 100:
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
                pump_on = images[::2]
                pump_off = images[1::2]
                if border_1 / border_2 < 100:
                    warnings.warn(
                        "low confidence in distnguishing pump on/off data",
                        UndistinguishableWarning,
                    )
            else:
                pump_on = images[1::2]
                pump_off = images[::2]
                if border_2 / border_1 < 100:
                    warnings.warn("low confidence in distnguishing pump on/off data")
        difference_mean = np.mean(pump_on - pump_off, axis=0)
        pump_on_mean = np.mean(pump_on, axis=0)
        pump_on_intensities = np.sum(pump_on*self.mask, axis=(1, 2))
        pump_off_mean = np.mean(pump_off, axis=0)
        pump_off_intensities = np.sum(pump_off*self.mask, axis=(1, 2))

        with h5py.File(dest, "w") as f:
            f.create_dataset("pump_on", data=pump_on_mean, **hdf5plugin.Bitshuffle())
            f.create_dataset("pump_on_sum_intensities", data=pump_on_intensities, **hdf5plugin.Bitshuffle())
            f.create_dataset("pump_off", data=pump_off_mean, **hdf5plugin.Bitshuffle())
            f.create_dataset("pump_off_sum_intensities", data=pump_off_intensities, **hdf5plugin.Bitshuffle())
            f.create_dataset(
                "difference", data=difference_mean, **hdf5plugin.Bitshuffle()
            )
