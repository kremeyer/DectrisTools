"""
module to quickly take a snapshot in .h5 format
"""
import warnings
from os import getcwd
from argparse import ArgumentParser
from time import sleep
from tqdm import tqdm
import nidaqmx
from nidaqmx.constants import VoltageUnits, Edge, AcquisitionType
from nidaqmx.stream_writers import AnalogSingleChannelWriter
import numpy as np
import ctypes
import threading
from uedinst.dectris import Quadro
from . import IP, PORT


warnings.simplefilter("ignore", ResourceWarning)
nidaq = ctypes.windll.nicaiu # load the DLL


class WaveformThread(threading.Thread):
    """
    This class performs the necessary initialization of the DAQ hardware and
    spawns a thread to handle playback of the signal.
    It takes as input arguments the waveform to play and the sample rate at which
    to play it.
    This will play an arbitrary-length waveform file.
    """
    def __init__( self, waveform, sampleRate, acq_type=AcquisitionType.CONTINUOUS):

        self.running = True
        self.sampleRate = sampleRate
        self.periodLength = len(waveform)
        self.data = np.zeros((self.periodLength,), dtype=np.float64)
        self.task = nidaqmx.Task()
        # convert waveform to a numpy array
        for i in range(self.periodLength):
            self.data[i] = waveform[i]
        # setup the DAQ hardware

        self.task.ao_channels.add_ao_voltage_chan(
                                   physical_channel="Dev1/ao0",
                                   name_to_assign_to_channel="",
                                   min_val=np.float64(-10.0),
                                   max_val=np.float64(10.0),
                                   units=VoltageUnits.VOLTS,
                                   custom_scale_name="")
        self.task.timing.cfg_samp_clk_timing(
                                rate=np.float64(self.sampleRate),
                                active_edge=Edge.RISING,
                                sample_mode=acq_type,
                               samps_per_chan = self.periodLength)
        self.Writer = AnalogSingleChannelWriter(self.task.out_stream, auto_start=True)
        self.Writer.write_many_sample(self.data, timeout=-1)
        threading.Thread.__init__( self )

    def CHK( self, err ):
        """a simple error checking routine for when calling lbirary directly"""
        if err < 0:
            buf_size = 100
            buf = ctypes.create_string_buffer( buf_size)
            nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
            raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))
        if err > 0:
            buf_size = 100
            buf = ctypes.create_string_buffer( buf_size)
            nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
            raise RuntimeError('nidaq generated warning %d: %s'%(err,repr(buf.value)))
    def start( self ):
        counter = 0
        self.task.start()
    def stop( self ):
        self.running = False
        self.task.stop()
        self.task.close()


def parse_args():
    parser = ArgumentParser(description='script to take a series of static diffraction images')
    parser.add_argument('n_images', type=int, help='number of images to take')
    parser.add_argument('exposure', type=float, help='exposure time per image in seconds')
    parser.add_argument('--dcu_ip', type=str, default=IP, help='DCU ip address')
    parser.add_argument('--dcu_port', type=int, default=PORT, help='DCU port')
    parser.add_argument('--shutter_port', type=str, default='COM20', help='com port of the shutter controller for the probe shutter')
    parser.add_argument('--savedir', type=str, help='save directory')
    parser.add_argument('--wait_time', type=float, default=0.2, help='time in s the shutter remains closed inbetween exposures')
    args = parser.parse_args()
    return args


def run(cmd_args):
    if cmd_args.savedir is None:
        cmd_args.savedir = getcwd()
    n = cmd_args.n_images
    exposure = cmd_args.exposure
    savedir = cmd_args.savedir
    wait_time = cmd_args.wait_time

    # prepare detector for experiment
    Q = Quadro(cmd_args.dcu_ip, cmd_args.dcu_port)
    if cmd_args.n_images <= 1000:
        Q.fw.nimages_per_file = 0
    else:
        Q.fw.nimages_per_file = 1000

    while Q.fw.mode == 'disabled':
        Q.fw.mode = 'enabled'
        sleep(0.1)
    Q.fw.clear()
    Q.trigger_mode = 'exte'
    Q.mon.buffer_size = 1
    Q.mon.mode = 'enabled'
    Q.mon.clear()

    Q.ntrigger = n
    Q.frame_time = exposure
    Q.count_time = exposure

    # start experiments
    Q.arm()
    try:
        for _ in tqdm(range(n)):
            x = np.append(4*np.ones(int(exposure*1e4)), 0)
            mythread = WaveformThread(x, 10000, acq_type=AcquisitionType.FINITE)
            sleep(exposure+wait_time+1)
            mythread.stop()
    except KeyboardInterrupt:
        pass

    sleep(3)
    for f in Q.fw.files:
        print(f'saving {savedir}/{f}')
        Q.fw.save(f, savedir)

    Q.disarm()
    Q.fw.mode = 'disabled'


if __name__ == '__main__':
    args = parse_args()
    run(args)
