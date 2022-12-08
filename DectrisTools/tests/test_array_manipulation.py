"""
tests for the array manipulation implemented in C
all functions are tested for data integriry against their python counterparts and their reference counting
"""
from sys import getrefcount
import numpy as np
from DectrisTools.lib.computation import masked_histogram as c_masked_histogram
from DectrisTools.lib.computation import masked_sum as c_masked_sum
from DectrisTools.lib.computation import normed_sum as c_normed_sum
from utils import IMAGES, MASK, NORM_VALUES


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


def __normed_sum_reference(images, norm_values):
    return np.sum(images / norm_values[:, None, None], axis=0)


def test_c_masked_histogram_integrity():
    reference = __masked_histogram_reference(IMAGES, MASK)
    assert (c_masked_histogram(IMAGES, MASK) == reference).all()


def test_c_masked_histogram_ref_counting():
    refcount_images = getrefcount(IMAGES)
    refcount_mask = getrefcount(MASK)
    c_masked_histogram(IMAGES, MASK)
    assert (getrefcount(IMAGES), getrefcount(MASK)) == (refcount_images, refcount_mask)


def test_c_masked_sum_integrity():
    reference = __masked_sum_reference(IMAGES, MASK)
    assert (c_masked_sum(IMAGES, MASK) == reference).all()


def test_c_masked_sum_ref_counting():
    refcount_images = getrefcount(IMAGES)
    refcount_mask = getrefcount(MASK)
    c_masked_sum(IMAGES, MASK)
    assert (getrefcount(IMAGES), getrefcount(MASK)) == (refcount_images, refcount_mask)


def test_c_normed_sum_integrity():
    reference = __normed_sum_reference(IMAGES, NORM_VALUES)
    assert (c_normed_sum(IMAGES, NORM_VALUES) == reference).all()


def test_c_normed_sum_ref_counting():
    refcount_images = getrefcount(IMAGES)
    refcount_norm_values = getrefcount(NORM_VALUES)
    c_normed_sum(IMAGES, NORM_VALUES)
    assert (getrefcount(IMAGES), getrefcount(NORM_VALUES)) == (refcount_images, refcount_norm_values)
