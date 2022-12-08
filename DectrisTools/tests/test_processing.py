"""
tests for the processing module
"""
from numpy import NaN, isnan
from ..lib.processing import slice_to_tuple, tuple_to_slice
from .utils import SLICES, TUPLES


def test_slice_to_tuple():
    for s, t in zip(SLICES, TUPLES):
        converted_tuple = slice_to_tuple(s)
        for converted_val, t_val in zip(converted_tuple, t):
            if isnan(t_val):
                # this is neccessary because (NaN == NaN) will yield False
                assert isnan(converted_val)
            else:
                assert converted_val == t_val


def test_tuple_to_slice():
    for t, s in zip(TUPLES, SLICES):
        assert tuple_to_slice(t) == s
