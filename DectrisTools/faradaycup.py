import numpy as np 
from argparse import ArgumentParser
from os import getcwd
from os.path import join
from uedinst.electrometer import Keithley6514
from scipy.constants import elementary_charge
from matplotlib import pyplot as plt


"""
Command line tool to measure relation between Faraday cup charge and wave plate angle in the Siwick lab
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--gpib", type=str, default=25, help="GPIB address")
    parser.add_argument("--savedir", type=str, default=getcwd() help="save directory")
    args = parser.parse_args()
    return args

def run(args):
    measure = True
    electrometer = Keithley6514(f'GPIB::{args[0]}')
    print('Electrometer connected)
    print('enter intergration time in seconds')
    aq_time = float(input())
    WP, charge, electrons = [], [], []
    while measure==True:
        print('enter waveplate angle to measure data point or X to save and exit:')
        user_in = input()
        if user_in is not 'X':
            WP.append(float(user_in))
            charge.append(electrometer.integrate('CHAR', aq_time))
            electrons.append(charge[-1]/(-elementary_charge)/(aq_time*1000))
            print(f'Readings for WP position {WP[-1]} are \n charge = {charge[-1]:.3e} C \n e- per shot = {electrons[-1]:.3e}')
        else:
            out = np.array([WP, charge, electrons])
            np.save(join(arg[1],'electrometer.npy'), out)
            measure=False
    print('Data saved to ' + str(join(arg[1],'electrometer.npy')))
    plt.(out[0],out[2], 'o')
    plt.xlabel('Waveplate angle')
    plt.ylabel('electrons per shot')
    plt.show()        

if __name__ = '__main__':
    args = parse_args()
    run(args)

