"""
Quadro module

this module collects classes used to connect to the DCU of the Dectris detector and make use of the detector itself
"""

import numpy as np
from .lib.DEigerClient import DEigerClient

# CONSTANTS
# these lists are read by the following classes to generate some of their methods
CONFIGS_READ = ['description',
                'detector_number',
                'eiger_fw_version',
                'software_version',
                'x_pixel_size',
                'y_pixel_size',
                'x_pixels_in_detector',
                'y_pixels_in_detector',
                'number_of_excluded_pixels',
                'bit_depth_image',
                'bit_depth_readout',
                'sensor_material',
                'sensor_thickness',
                ]
CONFIGS_WRTIE_NUM = ['beam_center_x',
                     'beam_center_y',
                     'count_time',
                     'frame_time',
                     'nimages',
                     'ntrigger',
                     'trigger_start_delay',
                     ]
STATS = ['state', 'time', 'humidity', 'temperature']
COMMANDS = ['initialize', 'arm', 'disarm', 'trigger', 'cancel', 'abort', 'check_connections', 'hv_reset']
FW_STATS = ['buffer_free', 'files', 'state']
FW_COMMANDS = ['clear']
MON_STATS = ['buffer_fill_level', 'dropped', 'error', 'monitor_image_number', 'next_image_number', 'state']
MON_COMMANDS = ['clear']


class QuadroError(Exception):
    """
    basic exception
    """
    pass


class FileWriter:
    """
    class for interaction with the filewriter subsystem of the Dectris DCU
    """
    def __init__(self, parent):
        self.parent = parent

        # read available command from constants defined above and create corresponding methods
        for command in FW_COMMANDS:
            setattr(self, command, self.__make_command_method(command))

    def __getattribute__(self, key):
        """
        if the requested attribute represents a state of the filewriter subsystem of the DCU, the api is called to
        return the state
        for all other attributes the default behaviour of object is mirrored
        """
        if key in FW_STATS:
            return self.parent.fileWriterStatus(key)['value']
        return object.__getattribute__(self, key)

    def __make_command_method(self, command):
        """
        method factory for commands that may be sent to the filewriter subsystem of the DCU
        """
        def command_method():
            self.parent.sendFileWriterCommand(command)

        command_method.__name__ = command
        return command_method

    def save(self, *args, **kwargs):
        self.parent.fileWriterSave(*args, **kwargs)

    """
    all methods below mirror the names of the variables found in the SIMPLON API manual
    """
    @property
    def image_nr_start(self):
        return self.parent.fileWriterConfig('image_nr_start')['value']

    @image_nr_start.setter
    def image_nr_start(self, i):
        if not isinstance(i, int):
            raise QuadroError(f'setting image_nr_start requires value of type int; not {type(i)}')
        self.parent.setFileWriterConfig('image_nr_start', i)

    @property
    def mode(self):
        return self.parent.fileWriterConfig('mode')['value']

    @mode.setter
    def mode(self, mode):
        if mode not in ['enabled', 'disabled']:
            raise QuadroError(f'setting mode requires "enabled" or "disabled"')
        self.parent.setFileWriterConfig('mode', mode)

    @property
    def name_pattern(self):
        return self.parent.fileWriterConfig('name_pattern')['value']

    @name_pattern.setter
    def name_pattern(self, pattern):
        if not isinstance(pattern, str):
            raise QuadroError(f'setting name_pattern requires value of type str; not {type(pattern)}')
        self.parent.setFileWriterConfig('name_pattern', pattern)

    @property
    def nimages_per_file(self):
        return self.parent.fileWriterConfig('nimages_per_file')['value']

    @nimages_per_file.setter
    def nimages_per_file(self, n):
        if not isinstance(n, int):
            raise QuadroError(f'setting nimages_per_file requires value of type int; not {type(n)}')
        if n < 0:
            raise QuadroError('setting nimages_per_file cannot be negative')
        self.parent.setFileWriterConfig('nimages_per_file', n)


class Monitor:
    """
    class for interaction with the monitor subsystem of the Dectris DCU
    """
    def __init__(self, parent):
        self.parent = parent

        # read available command from constants defined above and create corresponding methods
        for command in MON_COMMANDS:
            setattr(self, command, self.__make_command_method(command))

    def __getattribute__(self, key):
        """
        if the requested attribute represents a state of the monitor subsystem of the DCU, the api is called to
        return the state
        for all other attributes the default behaviour of object is mirrored
        """
        if key in MON_STATS:
            return self.parent.monitorStatus(key)['value']
        return object.__getattribute__(self, key)

    def __make_command_method(self, command):
        """
        method factory for commands that may be sent to the filewriter subsystem of the DCU
        """
        def command_method():
            self.parent.sendMonitorCommand(command)

        command_method.__name__ = command
        return command_method

    @property
    def last_image(self):
        """
        return the newest image in monitor
        """
        return self.parent.monitorImages('monitor')

    @property
    def first_image(self):
        """
        return the oldest image in monitor
        """
        return self.parent.monitorImages('next')

    @property
    def image_list(self):
        """
        return list of images in monitor
        """
        return self.parent.monitorImages(None)

    def save_last_image(self, path):
        """
        save the newest image in monitor to disk
        """
        self.parent.monitorSave('monitor', path)

    def save_first_image(self, path):
        """
        save the oldest image in monitor to disk
        """
        self.parent.monitorSave('next', path)

    """
    all methods below mirror the names of the variables found in the SIMPLON API manual
    """
    @property
    def buffer_size(self):
        return self.parent.monitorConfig('buffer_size')['value']

    @buffer_size.setter
    def buffer_size(self, n):
        if not isinstance(n, int):
            raise QuadroError(f'setting buffer_size requires value of type int; not {type(n)}')
        if n < 0:
            raise QuadroError('setting buffer_size cannot be negative')
        # TODO: ADD CHECK FOR MAX VALUE
        self.parent.setMonitorConfig('buffer_size', n)

    @property
    def discard_new(self):
        return self.parent.monitorConfig('discard_new')['value']

    @discard_new.setter
    def discard_new(self, state):
        if not isinstance(state, bool):
            raise QuadroError(f'setting discard new must be of type bool; not {type(state)}')
        self.parent.setMonitorConfig('discard_new', state)

    @property
    def mode(self):
        return self.parent.setMonitorConfig('mode')['value']

    @mode.setter
    def mode(self, mode):
        if mode not in ['enabled', 'disabled']:
            raise QuadroError(f'setting mode requires "enabled" or "disabled"')
        self.parent.setMonitorConfig('mode', mode)


class Quadro(DEigerClient):
    """
    main class for interacting with the Dectris DCU and the connected Detector

    all interactions with the detector itself are bound to self
    all interactions with the monitor subsystem are bound to self.mon
    all interactions with the filewriter subsystem are ound to self.fw
    the stream subsystem is yet to be added
    """
    def __init__(self, *args, **kwargs):
        """
        all arguments of the __init__ method are passed to the SIMPLON API DEigerClient class
        """
        super(Quadro, self).__init__(*args, **kwargs)

        # read available command from constants defined above and create corresponding methods
        for command in COMMANDS:
            setattr(self, command, self.__make_command_method(command))

        # creating instances of subsystems
        self.fw = FileWriter(parent=self)
        self.mon = Monitor(parent=self)

    def __getattribute__(self, key):
        """
        if the requested attribute represents a state of the detector, the api is called to return it
        for all other attributes the default behaviour of object is mirrored
        """
        if key in CONFIGS_READ or key in CONFIGS_WRTIE_NUM:
            return self.detectorConfig(key)['value']
        if key in STATS:
            return self.detectorStatus(key)['value']
        return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        """
        if the attribute to set represents a writeable numeric configuration of the detector, the api is called
        to write it
        for all other attributes the default behaviour of object is mirrored (or overwritten in the following)
        """
        if key in CONFIGS_WRTIE_NUM:
            if not isinstance(value, (int, float)):
                raise QuadroError(f'setting {key} requires value of type int or float; not {type(value)}')
            ans = self.detectorConfig('frame_time')
            if 'min' in ans.keys():
                low = ans['min']
            else:
                low = -np.inf
            if 'max' in ans.keys():
                high = ans['max']
            else:
                high = np.inf
            if value < low or value > high:
                raise QuadroError(f'setting out of range: {low} < {key} < {high}')
            self.setDetectorConfig(key, value)
        else:
            object.__setattr__(self, key, value)

    def __str__(self):
        """
        get basic detector info from print function
        """
        return f'Detector:                 {self.description}\n' \
               f'Serial:                   {self.detector_number}\n' \
               f'Eiger FW Version:         {self.eiger_fw_version}\n' \
               f'Decetor software Version: {self.software_version}\n' \
               f'Resolution:               {self.x_pixels_in_detector}x{self.y_pixels_in_detector}\n' \
               f'Pixel size:               {self.x_pixel_size * 1e6}x{self.y_pixel_size * 1e6} µm^2'

    def __make_command_method(self, command):
        """
        method factory for commands that may be sent to the detector
        """
        def command_method():
            self.sendDetectorCommand(command)

        command_method.__name__ = command
        return command_method

    """
    all methods below mirror the names of the variables found in the SIMPLON API manual
    """
    @property
    def auto_summation(self):
        return self.detectorConfig('auto_summation')['value']

    @auto_summation.setter
    def auto_summation(self, status):
        if not isinstance(status, bool):
            raise QuadroError('auto_summation property is expecting dtype "bool"')
        self.setDetectorConfig('auto_summation', status)

    @property
    def incident_energy(self):
        return self.detectorConfig('incident_energy')['value']

    @incident_energy.setter
    def incident_energy(self, e):
        allowed = np.array(self.detectorConfig('incident_energy')['allowed_values'])
        if not isinstance(e, (int, float)):
            raise QuadroError('incident energy property is expecting dtype "int"')
        if e not in allowed:
            e = float(allowed[np.argmin(abs(allowed - e))])
            print(f'WARNING: given incident energy is not allowed; rounding to nearest allowed energy {e:.0f}eV')
        self.setDetectorConfig('incident_energy', e)

    @property
    def trigger_mode(self):
        return self.detectorConfig('trigger_mode')['value']

    @trigger_mode.setter
    def trigger_mode(self, mode):
        allowed = self.detectorConfig('trigger_mode')['allowed_values']
        if mode not in allowed:
            raise QuadroError(f'trigger mode {mode} not allowed; use one of {allowed}')
        self.setDetectorConfig('trigger_mode', mode)
