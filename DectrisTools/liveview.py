import logging as log
from PyQt5 import QtWidgets
from argparse import ArgumentParser
from .ui.liveview import LiveViewUi


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ip', type=str, help='DCU ip address')
    parser.add_argument('port', type=int, help='DCU port')
    parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
    parser.add_argument('--update_interval', type=int, default=50, help='time between dectector image calls in ms')

    args = parser.parse_args()

    if args.verbose:
        log.basicConfig(format='[%(asctime)s] %(levelname)-8s | %(message)s', level='DEBUG', datefmt='%H:%M:%S')
    else:
        log.basicConfig(format='[%(asctime)s] %(levelname)-8s | %(message)s', level='WARNING', datefmt='%H:%M:%S')

    return args


def run():
    import sys

    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    ui = LiveViewUi(args)
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
