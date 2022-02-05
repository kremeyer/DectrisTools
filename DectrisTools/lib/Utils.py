from time import sleep
import logging as log
import io
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread
import numpy as np
from PIL import Image
from ..Quadro import Quadro


class LiveViewWidget(pg.ImageView):
    x_size = 0
    y_size = 0
    cursor_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        log.debug('initializing LiveView')
        super().__init__()
        pg.setConfigOption('background', (240, 240, 240))
        pg.setConfigOption('foreground', 'k')
        self.setParent(parent)
        self.setPredefinedGradient('inferno')
        self.setLevels(0, 2**16)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        self.proxy = pg.SignalProxy(self.scene.sigMouseMoved,
                                    rateLimit=60, slot=self.__callback_move)

    @pyqtSlot(tuple)
    def __callback_move(self, evt):
        """
        callback function for mouse movement on image
        -> triggers status bar update
        """
        qpoint = self.view.mapSceneToView(evt[0])
        x = int(qpoint.x())
        y = int(qpoint.y())
        if x < 0 or x >= self.x_size:
            self.cursor_changed.emit((np.NaN, np.NaN))
            return
        if y < 0 or y >= self.y_size:
            self.cursor_changed.emit((np.NaN, np.NaN))
            return
        self.cursor_changed.emit((x, y))


class DectrisImageGrabber(QObject):
    image_ready = pyqtSignal(np.ndarray)
    exposure_triggered = pyqtSignal()
    connected = False

    def __init__(self, ip, port, trigger_mode='ints', exposure=0.3):
        super().__init__()

        self.Q = Quadro(ip, port)
        try:
            _ = self.Q.state
            self.connected = True
            log.info('DectrisImageGrabber successfully connected to detector')
            log.info(self.Q)
        except OSError:
            log.warning('DectrisImageGrabber could not establish connection to detector')

        if self.connected:
            if self.Q.state == 'na':
                log.warning('Detector need to be initialized, that may take a while...')
                self.Q.initialize()
            self.Q.mon.clear()
            self.Q.fw.clear()

            self.Q.fw.mode = 'disabled'
            self.Q.mon.mode = 'enabled'
            self.Q.incident_energy = 1e5
            self.Q.count_time = exposure
            self.Q.frame_time = exposure
            self.Q.trigger_mode = trigger_mode

        self.image_grabber_thread = QThread()
        self.moveToThread(self.image_grabber_thread)
        self.image_grabber_thread.started.connect(self.__get_image)

    def __del__(self):
        if self.connected:
            self.Q.mon.clear()
            self.Q.disarm()

    @pyqtSlot()
    def __get_image(self):

        log.debug(f'started image_grabber_thread {self.image_grabber_thread.currentThread()}')

        if self.connected:
            self.Q.arm()
            self.Q.trigger()
            self.exposure_triggered.emit()
            while not self.Q.state == 'idle':
                sleep(0.05)
            self.Q.disarm()
            while not self.Q.mon.image_list:
                sleep(0.05)
            # image comes as a file-like object in tif format
            self.image_ready.emit(np.array(Image.open(io.BytesIO(self.Q.mon.last_image))))
            self.Q.mon.clear()
        else:
            self.exposure_triggered.emit()
            sleep(5)
            self.image_ready.emit(np.random.rand(512, 512) * 2**16)

        self.image_grabber_thread.quit()
        log.debug(f'quit image_grabber_thread {self.image_grabber_thread.currentThread()}')


class DectrisStatusGrabber(QObject):
    status_ready = pyqtSignal(dict)
    connected = False

    def __init__(self, ip, port):
        super().__init__()

        self.Q = Quadro(ip, port)
        try:
            _ = self.Q.state
            self.connected = True
            log.info('DectrisStatusGrabber successfully connected to detector')
            log.info(self.Q)
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
                                    'trigger_mode': self.Q.trigger_mode, 'exposure': self.Q.frame_time})
        else:
            self.status_ready.emit({'quadro': None, 'fw': None, 'mon': None, 'trigger_mode': None, 'exposure': None})
        self.status_grabber_thread.quit()
        log.debug(f'quit status_grabber_thread {self.status_grabber_thread.currentThread()}')


class ExposureProgressWorker(QObject):
    advance_progress_bar = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.progress_thread = QThread()
        self.moveToThread(self.progress_thread)
        self.progress_thread.started.connect(self.__start_progress)

    @pyqtSlot()
    def __start_progress(self):
        while True:
            self.advance_progress_bar.emit()
            sleep(0.01)


def interrupt_liveview(f):
    def wrapper(self):
        log.debug('stopping liveview')
        self.image_timer.stop()
        log.debug('waiting for image grabing thread to finish')
        # wait 2*exposure time for detector to finish; otherwise abort
        if self.dectris_image_grabber.connected:
            for _ in range(int(self.dectris_image_grabber.Q.frame_time)*1000):
                if self.dectris_image_grabber.image_grabber_thread.isFinished():
                    break
                sleep(0.002)
            if not self.dectris_image_grabber.image_grabber_thread.isFinished():
                log.warning('image grabbing thread does not seem to finish, aborting acquisition')
                if self.dectris_image_grabber.connected:
                    self.dectris_image_grabber.Q.abort()
        f(self)
        log.debug('restarting liveview')
        self.image_timer.start(self.update_interval)
    return wrapper
