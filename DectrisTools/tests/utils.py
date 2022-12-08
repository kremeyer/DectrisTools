"""
variables required for testing functionality of the package
"""
from os.path import join
import numpy as np
from DectrisTools import get_base_path

STACKSIZE = 1_000
IMG_SHAPE = (512, 512)

IMAGES = np.random.randint(0, 100, (STACKSIZE, IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.uint16)
MASK = np.random.randint(0, 1, IMG_SHAPE, dtype=np.uint16)
NORM_VALUES = np.random.random(STACKSIZE).astype(np.float32)

SLICES = [
    slice(0, 100, 2),
    slice(10, 100, -2),
    slice(10, 100, 2),
    slice(10, -100, -2),
    slice(10, -100, 2),
    slice(-10, 100, -2),
    slice(-10, 100, 2),
    slice(-10, -100, -2),
    slice(-10, -100, 2),
    slice(None, -100, 2),
    slice(-10, None, 2),
    slice(-10, -100, None),
]

TUPLES = [
    (0, 100, 2),
    (10, 100, -2),
    (10, 100, 2),
    (10, -100, -2),
    (10, -100, 2),
    (-10, 100, -2),
    (-10, 100, 2),
    (-10, -100, -2),
    (-10, -100, 2),
    (np.NaN, -100, 2),
    (-10, np.NaN, 2),
    (-10, -100, np.NaN),
]

LOGFILE = join(get_base_path(), 'tests', 'experiment.log')
