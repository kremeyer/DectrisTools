"""
module to continiously print the current state of the detector and it's subsystem to the terminal
"""
import os
from argparse import ArgumentParser
from time import sleep
from uedinst.dectris import Quadro
from . import IP, PORT


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ip", type=str, default=IP, help="DCU ip address")
    parser.add_argument("--port", type=int, default=PORT, help="DCU port")
    parser.add_argument(
        "--update_interval",
        type=int,
        default=250,
        help="time between dectector image calls in ms",
    )
    args = parser.parse_args()
    return args


def clear_output():
    if os.name in ("nt", "dos"):
        os.system("cls")
    else:
        os.system("clear")


def run():
    args = parse_args()
    q = Quadro(args.ip, args.port)

    while True:
        print(q)
        print(f"detector state:           {q.state}")
        print(f"monitor state:            {q.mon.state}")
        print(f"filewriter state:         {q.fw.state}")
        sleep(args.update_interval / 1000)
        clear_output()


if __name__ == "__main__":
    run()
