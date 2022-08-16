"""
collection of helper classes and functions
"""
from time import sleep
import logging as log
import io
from collections import deque
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtWidgets import QAction, QMenu
import numpy as np
import pyqtgraph as pg
from PIL import Image
from uedinst.dectris import Quadro


def monitor_to_array(bytestring):
    """
    image comes as a file-like object in tif format and is returned as a np.ndarray
    """
    return np.rot90(np.array(Image.open(io.BytesIO(bytestring))), k=3)


class DectrisImageGrabber(QObject):
    """
    class capable of setting the collecting images from the detector in a non-blocking fashion
    """
    image_ready = pyqtSignal(np.ndarray)
    exposure_triggered = pyqtSignal()
    connected = False

    def __init__(self, ip, port, trigger_mode='ints', exposure=0.3):
        super().__init__()

        self.Q = Quadro(ip, port)
        try:
            _ = self.Q.state
            self.connected = True
            log.info(f'DectrisImageGrabber successfully connected to detector\n{self.Q}')
        except OSError:
            log.warning('DectrisImageGrabber could not establish connection to detector')

        # prepare the hardware for taking images
        if self.connected:
            if self.Q.state == 'na':
                log.warning('Detector needs to be initialized, that may take a while...')
                self.Q.initialize()
            self.Q.mon.clear()
            self.Q.fw.clear()
            self.Q.fw.mode = 'disabled'
            self.Q.mon.mode = 'enabled'
            self.Q.incident_energy = 1e5
            self.Q.count_time = exposure
            self.Q.frame_time = exposure
            self.Q.trigger_mode = trigger_mode
            self.Q.ntrigger = 1

        self.image_grabber_thread = QThread()
        self.moveToThread(self.image_grabber_thread)
        self.image_grabber_thread.started.connect(self.__get_image)

    def __del__(self):
        if self.connected:
            self.Q.mon.clear()
            self.Q.abort()

    @pyqtSlot()
    def __get_image(self):
        """
        image collection method
        """
        log.debug(f'started image_grabber_thread {self.image_grabber_thread.currentThread()}')
        if self.connected:
            self.Q.arm()
            # logic for different trigger modes
            if self.Q.trigger_mode == 'ints':
                self.exposure_triggered.emit()
                self.wait_for_state('idle')
                self.Q.trigger()
                self.wait_for_state('idle', False)
                self.Q.disarm()
            if self.Q.trigger_mode == 'exts':
                self.wait_for_state('ready')
                self.exposure_triggered.emit()
                self.wait_for_state('acquire')
            # wait until images appears in monitor
            while not self.Q.mon.image_list:
                if self.image_grabber_thread.isInterruptionRequested():
                    self.image_grabber_thread.quit()
                    return
                sleep(0.05)
            self.image_ready.emit(monitor_to_array(self.Q.mon.last_image))
            self.Q.mon.clear()
        else:
            # simulated image for @home use
            self.exposure_triggered.emit()
            sleep(1)
            x = np.linspace(-10, 10, 512)
            xs, ys = np.meshgrid(x, x)
            img = 5e4 * ((np.cos(np.hypot(xs, ys)) / (np.hypot(xs, ys)+1) * np.random.normal(1, 0.4, (512, 512))) + 0.3)
            self.image_ready.emit(img)

        self.image_grabber_thread.quit()
        log.debug(f'quit image_grabber_thread {self.image_grabber_thread.currentThread()}')

    def wait_for_state(self, state_name, logic=True):
        """
        making sure waiting for the detector to enter or leave a state is not blocking the interruption of the thread
        """
        log.debug(f'waiting for state: {state_name} to be {logic}')
        if logic:
            while self.Q.state == state_name:
                if self.image_grabber_thread.isInterruptionRequested():
                    self.image_grabber_thread.quit()
                    return
                sleep(0.05)
            return
        while self.Q.state != state_name:
            if self.image_grabber_thread.isInterruptionRequested():
                self.image_grabber_thread.quit()
                return
            sleep(0.05)


class DectrisStatusGrabber(QObject):
    """
    class for continiously retrieving status information from the DCU
    """
    status_ready = pyqtSignal(dict)
    connected = False

    def __init__(self, ip, port):
        super().__init__()

        self.Q = Quadro(ip, port)
        try:
            _ = self.Q.state
            self.connected = True
            log.info('DectrisStatusGrabber successfully connected to detector')
        except OSError:
            log.warning('DectrisStatusGrabber could not establish connection to detector')

        self.status_grabber_thread = QThread()
        self.moveToThread(self.status_grabber_thread)
        self.status_grabber_thread.started.connect(self.__get_status)

    @pyqtSlot()
    def __get_status(self):
        log.debug(f'started status_grabber_thread {self.status_grabber_thread.currentThread()}')
        if self.connected:
            self.status_ready.emit({'quadro': self.Q.state, 'fw': self.Q.fw.state, 'mon': self.Q.mon.state,
                                    'trigger_mode': self.Q.trigger_mode, 'exposure': self.Q.frame_time,
                                    'counting_mode': self.Q.counting_mode})
        else:
            self.status_ready.emit({'quadro': None, 'fw': None, 'mon': None, 'trigger_mode': None, 'exposure': None, 'counting_mode': None})
        self.status_grabber_thread.quit()
        log.debug(f'quit status_grabber_thread {self.status_grabber_thread.currentThread()}')


def interrupt_acquisition(f):
    """
    decorator interrupting/resuming image acquisition before/after function call
    """
    def wrapper(self):
        log.debug('stopping liveview')
        self.image_timer.stop()
        if self.dectris_image_grabber.connected:
            if not self.dectris_image_grabber.image_grabber_thread.isFinished():
                log.debug('aborting acquisition')
                self.dectris_image_grabber.Q.abort()
                self.dectris_image_grabber.image_grabber_thread.requestInterruption()
                self.dectris_image_grabber.image_grabber_thread.wait()
                self.dectris_image_grabber.Q.mon.clear()
        f(self)
        if not self.actionStop.isChecked():
            log.debug('restarting liveview')
            self.image_timer.start(self.update_interval)
    return wrapper


class RectROI(pg.RectROI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = pg.GraphicsLayoutWidget(title='ROI integrated intensity')
        self.plot = self.win.addPlot()
        self.plot.axes['left']['item'].setLabel('mean intensity')
        self.plot.axes['bottom']['item'].setLabel('image index')
        self.curve = self.plot.plot()
        self.last_means = deque(maxlen=30)

    def getMenu(self):
        if self.menu is None:
            self.menu = QMenu()
            self.menu.setTitle("ROI")
            rem_act = QAction("Remove ROI", self.menu)
            rem_act.triggered.connect(self.removeClicked)
            self.menu.addAction(rem_act)
            self.menu.rem_act = rem_act
            history_act = QAction("Show mean history", self.menu)
            history_act.triggered.connect(self.integral_plot_clicked)
            self.menu.addAction(history_act)
            self.menu.history_act = history_act
        self.menu.setEnabled(self.contextMenuEnabled())
        return self.menu

    def integral_plot_clicked(self):
        self.win.show()

    def add_mean(self, data, img):
        self.last_means.append(self.getArrayRegion(data, img).mean())
        self.curve.setData(x=range(-len(self.last_means)+1, 1), y=self.last_means)
