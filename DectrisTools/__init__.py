import os


VERSION = "0.5"
IP = "fe80::4ed9:8fff:feca:a8f9"
PORT = 80

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_base_path():
    """
    returns package base dir
    """
    return os.path.dirname(__file__)
