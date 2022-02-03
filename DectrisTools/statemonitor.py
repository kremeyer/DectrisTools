import os
import logging as log
from argparse import ArgumentParser
from time import sleep
from .Quadro import Quadro


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ip', type=str, help='DCU ip address')
    parser.add_argument('port', type=int, help='DCU port')
    parser.add_argument('--update_interval', type=int, default=250, help='time between dectector image calls in ms')
    args = parser.parse_args()
    return args


def clear_output():
    if os.name in ('nt', 'dos'):
        os.system('cls')
    else:
        os.system('clear')


def run():
    args = parse_args()
    Q = Quadro(args.ip, args.port)

    while True:
        print(Q)
        print(f'detector state:           {Q.state}')
        print(f'monitor state:            {Q.mon.state}')
        print(f'filewriter state:         {Q.fw.state}')
        sleep(args.update_interval/1000)
        clear_output()


if __name__ == '__main__':
    run()
