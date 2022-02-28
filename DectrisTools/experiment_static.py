"""
module to quickly take a snapshot in .h5 format
"""
import warnings
from os import getcwd, path
from argparse import ArgumentParser
from tqdm import tqdm
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from uedinst.dectris import Quadro
from uedinst.shutter import SC10Shutter
from . import IP, PORT
from .lib.Utils import monitor_to_array


warnings.simplefilter("ignore", ResourceWarning)


def parse_args():
    parser = ArgumentParser(description='script to take a series of static diffraction images')
    parser.add_argument('n_images', type=int, help='number of images to take')
    parser.add_argument('exposure', type=float, help='exposure time per image in seconds')
    parser.add_argument('--dcu_ip', type=str, default=IP, help='DCU ip address')
    parser.add_argument('--dcu_port', type=int, default=PORT, help='DCU port')
    parser.add_argument('--shutter_port', type=str, default='COM20', help='com port of the shutter controller for the probe shutter')
    parser.add_argument('--savedir', type=str, help='save directory')
    parser.add_argument('--wait_time', type=float, default=0.1, help='time in s the shutter remains closed inbetween exposures')
    args = parser.parse_args()
    return args


def run(cmd_args):
    if cmd_args.savedir is None:
        cmd_args.savedir = getcwd()
    n = cmd_args.n_images
    exposure = cmd_args.exposure
    savedir = cmd_args.savedir

    # prepare shutter for experiment
    S = SC10Shutter(args.shutter_port)
    S.set_operating_mode('single')
    S.set_trigger_mode('internal')
    S.set_open_time(args.exposure*1000)
    S.set_close_time(args.wait_time*1000)

    # prepare detector for experiment
    Q = Quadro(cmd_args.dcu_ip, cmd_args.dcu_port)
    if cmd_args.n_images <= 1000:
        Q.fw.nimages_per_file = 0
    else:
        Q.fw.nimages_per_file = 1000

    Q.fw.clear()
    Q.trigger_mode = 'exte'
    Q.fw.mode = 'enabled'
    Q.mon.buffer_size = 1
    Q.mon.mode = 'enabled'
    Q.mon.clear()

    Q.ntrigger = n
    Q.frame_time = exposure
    Q.count_time = exposure

    # start experiments
    Q.arm()
    for _ in tqdm(range(n)):
        S.enable(True)
        Q.mon.save_last_image(path.join(savedir, 'last_img.tif'))

    for f in Q.fw.files:
         print(f'saving {savedir}/{f}')
         Q.fw.save(f, savedir)
    
    Q.disarm()
    Q.fw.mode = 'disabled'


if __name__ == '__main__':
    args = parse_args()
    run(args)
