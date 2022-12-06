"""
module to quickly take a snapshot in .h5 format
"""
import warnings
from os import getcwd, path
from argparse import ArgumentParser
from time import sleep
from tqdm import tqdm
from PIL import Image
from uedinst.dectris import Quadro
from uedinst.shutter import SC10Shutter
from uedinst import InstrumentException
from . import IP, PORT


warnings.simplefilter("ignore", ResourceWarning)


def parse_args():
    parser = ArgumentParser(description="script to take a series of static diffraction images")
    parser.add_argument("--dcu_ip", type=str, default=IP, help="DCU ip address")
    parser.add_argument("--dcu_port", type=int, default=PORT, help="DCU port")
    parser.add_argument(
        "--shutter_port",
        type=str,
        default="COM20",
        help="com port of the shutter controller for the probe shutter",
    )
    parser.add_argument("--savedir", type=str, help="save directory")
    args = parser.parse_args()
    return args


def run(cmd_args):
    if cmd_args.savedir is None:
        cmd_args.savedir = getcwd()
    savedir = cmd_args.savedir

    # prepare shutter for experiment
    # S = SC10Shutter(args.shutter_port)
    # S.set_operating_mode('manual')
    # S.enable(True)

    # prepare detector for experiment
    Q = Quadro(cmd_args.dcu_ip, cmd_args.dcu_port)
    Q.fw.nimages_per_file = 0
    Q.fw.mode = "enabled"
    Q.fw.clear()
    Q.trigger_mode = "exte"
    Q.mon.mode = "disabled"

    Q.ntrigger = int(15e3)
    Q.frame_time = 0.001
    Q.count_time = 0.00025

    # start experiments
    Q.arm()
    while Q.state != "idle":
        sleep(0.1)
    sleep(2)

    for f in Q.fw.files:
        print(f"saving {savedir}/{f}")
        Q.fw.save(f, savedir)
    Q.disarm()
    # sleep( 5 )
    # mythread.Writer.write_one_sample(0)
    # mythread.stop()
    # S.enable(False)


if __name__ == "__main__":
    args = parse_args()
    run(args)
