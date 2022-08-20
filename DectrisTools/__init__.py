import os

# imports for top-level access
from .lib.processing import SingleShotDataset, SingleShotProcessor

VERSION = "0.1"
IP = "fe80::4ed9:8fff:feca:a8f9"
PORT = 80


def get_base_path():
    """
    returns package base dir
    """
    return os.path.dirname(__file__)
