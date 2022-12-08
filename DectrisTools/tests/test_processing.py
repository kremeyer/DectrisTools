"""
tests for the processing module
"""
from numpy import isnan
from DectrisTools.lib.processing import slice_to_tuple, tuple_to_slice, filenames_from_logfile, \
    timestamps_from_logfile, delay_from_fname
from .utils import SLICES, TUPLES, LOGFILE


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


def test_filenames_from_logfile():
    n_files = 4380
    parsed_files = filenames_from_logfile(LOGFILE)

    assert len(parsed_files) == n_files  # not very bulletproof


def test_timestamps_from_logfile():
    n_files = 4380
    parsed_timestamps = timestamps_from_logfile(LOGFILE)

    assert len(parsed_timestamps) == n_files
    assert sorted(parsed_timestamps) == parsed_timestamps


def test_delay_from_filename():
    reference_delays = [-46., -45.5, -45, -44.5, -44, -43.9, -43.8, -43.7, -43.6, -43.5, -43.4, -43.3,
                        -43.2, -43.1, -43, -42.9, -42.8, -42.7, -42.6, -42.5, -42.4, -42.3, -42.2, -42.1,
                        -42, -41.9, -41.8, -41.7, -41.6, -41.5, -41.4, -41.3, -41.2, -41.1, -41, -40.9,
                        -40.8, -40.7, -40.6, -40.5, -40.4, -40.3, -40.2, -40.1, -40, -39.9, -39.8, -39.7,
                        -39.6, -39.5, -39.4, -39.3, -39.2, -39.1, -39, -38.5, -38, -37.5, -37, -36.5]
    parsed_files = filenames_from_logfile(LOGFILE)
    parsed_delays = [delay_from_fname(str(f)) for f in parsed_files]

    assert sorted(set(parsed_delays)) == reference_delays
