from sys import getrefcount
import numpy as np
from DectrisTools.lib.computation import masked_histogram as c_masked_histogram
from DectrisTools.lib.computation import masked_sum as c_masked_sum
from DectrisTools.lib.computation import normed_stack as c_normed_stack


STACKSIZE = 1_000
IMG_SHAPE = (512, 512)
TEST_IMAGES = np.random.randint(0, 100, (STACKSIZE, IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.uint16)
TEST_MASK = np.random.randint(0, 1, IMG_SHAPE, dtype=np.uint16)
TEST_NORM_VALUES = np.random.random(STACKSIZE).astype(np.float32)


def __masked_histogram_reference(images, mask):
    histogram = np.zeros(2**16)
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            for k in range(images.shape[2]):
                if mask[j, k]:
                    histogram[images[i, j, k]] += 1
    return histogram


def __masked_sum_reference(images, mask):
    return np.sum(images * mask[None, :, :], axis=(1, 2))


def __normed_stack_reference(images, norm_values):
    return images / norm_values[:, None, None]


def test_c_masked_histogram_integrity():
    reference = __masked_histogram_reference(TEST_IMAGES, TEST_MASK)
    assert (c_masked_histogram(TEST_IMAGES, TEST_MASK) == reference).all()


def test_c_masked_histogram_ref_counting():
    refcount_images = getrefcount(TEST_IMAGES)
    refcount_mask = getrefcount(TEST_MASK)
    c_masked_histogram(TEST_IMAGES, TEST_MASK)
    assert (getrefcount(TEST_IMAGES), getrefcount(TEST_MASK)) == (refcount_images, refcount_mask)


def test_c_masked_sum_integrity():
    reference = __masked_sum_reference(TEST_IMAGES, TEST_MASK)
    assert (c_masked_sum(TEST_IMAGES, TEST_MASK) == reference).all()


def test_c_masked_sum_ref_counting():
    refcount_images = getrefcount(TEST_IMAGES)
    refcount_mask = getrefcount(TEST_MASK)
    c_masked_sum(TEST_IMAGES, TEST_MASK)
    assert (getrefcount(TEST_IMAGES), getrefcount(TEST_MASK)) == (refcount_images, refcount_mask)


def test_c_normed_stack_integrity():
    reference = __normed_stack_reference(TEST_IMAGES, TEST_NORM_VALUES)
    assert (c_normed_stack(TEST_IMAGES, TEST_NORM_VALUES) == reference).all()


def test_c_normed_stack_ref_counting():
    refcount_images = getrefcount(TEST_IMAGES)
    refcount_norm_values = getrefcount(TEST_NORM_VALUES)
    c_normed_stack(TEST_IMAGES, TEST_NORM_VALUES)
    assert (getrefcount(TEST_IMAGES), getrefcount(TEST_NORM_VALUES)) == (refcount_images, refcount_norm_values)
