from time import sleep
import logging as log
import io
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, QObject, QThread
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

    def __init__(self, ip, port):
        super().__init__()

        self.Q = Quadro(ip, port)
        if self.Q.state == 'na':
            log.warning('Detector need to be initialized, that may take a while...')
            self.Q.initialize()
        self.Q.mon.clear()
        self.Q.fw.clear()

        self.Q.fw.mode = 'disabled'
        self.Q.mon.mode = 'enabled'
        self.Q.incident_energy = 1e5
        self.Q.count_time = 10
        self.Q.frame_time = 10
        self.Q.n_images = 1
        self.Q.n_trigger = 1
        # self.Q.trigger_mode = 'exte'
        self.Q.trigger_mode = 'ints'

        self.image_grabber_thread = QThread()
        self.moveToThread(self.image_grabber_thread)
        self.image_grabber_thread.started.connect(self.__get_image)

    def __del__(self):
        self.Q.mon.clear()
        self.Q.disarm()

    def __get_image(self):
        log.debug(f'started image_grabber_thread {self.image_grabber_thread.currentThread()}')
        # sleep(1)
        # self.image_ready.emit(np.random.rand(512, 512) * 2**16)

        self.Q.arm()
        self.Q.trigger()
        while not self.Q.state == 'idle':
            sleep(0.05)
        self.Q.disarm()
        while not self.Q.mon.image_list:
            sleep(0.05)
        # image comes as a file-like object in tif format
        self.image_ready.emit(np.array(Image.open(io.BytesIO(self.Q.mon.last_image))))
        self.Q.mon.clear()

        self.image_grabber_thread.quit()
        log.debug(f'quit image_grabber_thread {self.image_grabber_thread.currentThread()}')


class DectrisStatusGrabber(QObject):
    status_ready = pyqtSignal(dict)

    def __init__(self, ip, port):
        super().__init__()

        self.Q = Quadro(ip, port)

        self.status_grabber_thread = QThread()
        self.moveToThread(self.status_grabber_thread)
        self.status_grabber_thread.started.connect(self.__get_status)

    def __get_status(self):
        return {'quadro': self.Q.state, 'fw': self.Q.fw.state, 'mon': self.Q.mon.state}
