"""
module to quickly take a snapshot in .h5 format
"""
from time import sleep
import warnings
from os import getcwd
from argparse import ArgumentParser
from uedinst.dectris import Quadro
from . import IP, PORT

warnings.simplefilter("ignore", ResourceWarning)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ip', type=str, default=IP, help='DCU ip address')
    parser.add_argument('--port', type=int, default=PORT, help='DCU port')
    parser.add_argument('--savedir', type=str, help='save directory')
    args = parser.parse_args()
    return args


def run(cmd_args):
    if cmd_args.savedir is None:
        cmd_args.savedir = getcwd()

    Q = Quadro(cmd_args.ip, cmd_args.port)

    old_n_imgs = Q.fw.nimages_per_file
    Q.fw.nimages_per_file = 0
    Q.fw.clear()
    Q.fw.mode = 'enabled'
    while not Q.fw.files:
        try:
            sleep(0.05)
        except KeyboardInterrupt:
            Q.fw.nimages_per_file = old_n_imgs
            Q.fw.mode = 'disabled'
            break
    for f in Q.fw.files:
        print(f'saving {cmd_args.savedir}/{f}')
        Q.fw.save(f, cmd_args.savedir)
    
    Q.fw.nimages_per_file = old_n_imgs
    Q.fw.mode = 'disabled'


if __name__ == '__main__':
    args = parse_args()
    run(args)
