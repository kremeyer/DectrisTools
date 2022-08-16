import warnings
from argparse import ArgumentParser
from random import shuffle
from time import sleep, time
from datetime import datetime
from tqdm import tqdm
from os import rename, path, getcwd, mkdir
import numpy as np
from uedinst.dectris import Quadro
from uedinst.shutter import SC10Shutter
from uedinst import ILS250PP
from . import IP, PORT


warnings.simplefilter("ignore", ResourceWarning)

DIR_PUMP_OFF = "pump_off"
DIR_LASER_BG = "laser_background"
T0_POS = 27.1083


def parse_args():
    parser = ArgumentParser(description="script to take single shot experiment")
    parser.add_argument("--dcu_ip", type=str, default=IP, help="DCU ip address")
    parser.add_argument("--dcu_port", type=int, default=PORT, help="DCU port")
    parser.add_argument(
        "--pump_shutter_port",
        type=str,
        default="COM22",
        help="com port of the shutter controller for the pump shutter",
    )
    parser.add_argument(
        "--probe_shutter_port",
        type=str,
        default="COM20",
        help="com port of the shutter controller for the probe shutter",
    )
    parser.add_argument(
        "--delay_stage_ip",
        type=str,
        default="192.168.254.254",
        help="ip address of the delay stage",
    )
    parser.add_argument("--savedir", type=str, help="save directory")
    parser.add_argument("--n_scans", type=int, help="number of scans")
    parser.add_argument(
        "--images_per_datapoint",
        type=int,
        default=30_000,
        help="number of images per datapoint",
    )
    parser.add_argument("--delays", type=str)
    args = parser.parse_args()
    return args


def parse_timedelays(time_str):
    # shamelessly stolen from faraday
    time_str = str(time_str)
    elements = time_str.split(",")
    if not elements:
        return []
    timedelays = []

    # Two possibilities : floats or ranges
    # Either elem = float
    # or     elem = start:step:stop
    for elem in elements:
        try:
            fl = float(elem)
            timedelays.append(fl)
        except ValueError:
            try:
                start, step, stop = tuple(map(float, elem.split(":")))
                fl = np.round(np.arange(start, stop, step), 3).tolist()
                timedelays.extend(fl)
            except:
                return []

    # Round timedelays down to the femtosecond
    timedelays = map(lambda n: round(n, 3), timedelays)

    return list(sorted(timedelays))


def acquire_image_series(detector, savedir, scandir, filename):
    detector.arm()
    sleep(0.1)
    while detector.state != "idle":
        sleep(0.05)
    sleep(0.5)
    for f in detector.fw.files:
        detector.fw.save(f, savedir)
        rename(path.join(savedir, f), path.join(savedir, scandir, filename))
    detector.fw.clear()
    detector.disarm()


def fmt_log(message):
    return f"{datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')} | {message}\n"


def run(cmd_args):
    if cmd_args.savedir is None:
        cmd_args.savedir = getcwd()
    savedir = cmd_args.savedir
    delays = parse_timedelays(cmd_args.delays)

    # prepare hardware for experiment
    Q = Quadro(cmd_args.dcu_ip, cmd_args.dcu_port)

    Q.fw.nimages_per_file = 0
    Q.fw.mode = "enabled"
    Q.fw.clear()
    Q.trigger_mode = "exte"
    Q.mon.mode = "disabled"

    Q.ntrigger = cmd_args.images_per_datapoint
    Q.frame_time = 0.001
    Q.count_time = 0.00025

    s_pump = SC10Shutter(args.pump_shutter_port)
    s_pump.set_operating_mode("manual")
    s_probe = SC10Shutter(args.probe_shutter_port)
    s_probe.set_operating_mode("manual")

    delay_stage = ILS250PP(cmd_args.delay_stage_ip)

    # start experiment
    logfile = open(path.join(savedir, "experiment.log"), "w")
    logfile.write(
        fmt_log(
            f"starting experiment with {cmd_args.n_scans} scans at {len(delays)} delays, each image series contains {cmd_args.images_per_datapoint} images"
        )
    )
    mkdir(path.join(savedir, DIR_LASER_BG))
    mkdir(path.join(savedir, DIR_PUMP_OFF))
    for i in tqdm(range(cmd_args.n_scans), desc="scans"):
        s_pump.enable(True)
        s_probe.enable(False)
        acquire_image_series(
            Q, savedir, DIR_LASER_BG, f"laser_bg_epoch_{time():010.0f}s.h5"
        )
        logfile.write(fmt_log("laser background image series acquired"))
        s_pump.enable(False)
        s_probe.enable(True)
        acquire_image_series(
            Q, savedir, DIR_PUMP_OFF, f"pump_off_epoch_{time():010.0f}s.h5"
        )
        logfile.write(fmt_log("pump off image series acquired"))
        s_pump.enable(True)

        scandir = f"scan_{i+1:04d}"
        mkdir(path.join(savedir, scandir))
        shuffle(delays)
        for delay in tqdm(delays, leave=False, desc="delay steps"):
            filename = f"pumpon_{delay:+010.3f}ps.h5"

            delay_stage.absolute_time(delay, T0_POS)
            delay_stage._wait_end_of_move()
            acquire_image_series(Q, savedir, scandir, filename)
            logfile.write(
                fmt_log(
                    f"pump on image series acquired at scan {i+1} and time-delay {delay:.1f}ps"
                )
            )

    s_pump.enable(False)
    s_probe.enable(False)
    logfile.write(fmt_log("EXPERIMENT COMPLETE"))
    logfile.close()
    print("🍻")


if __name__ == "__main__":
    # TEST COMMAND:
    # python -m DectrisTools.single_shot_experiment --savedir="D:\Data\Tests\single_shot_exp" --images_per_datapoint=1000 --n_scans=3 --delays="0:1:5"
    args = parse_args()
    run(args)
