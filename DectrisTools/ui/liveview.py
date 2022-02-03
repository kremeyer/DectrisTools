from os import path
from time import sleep
import logging as log
import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from ..lib.Utils import DectrisImageGrabber
from .. import get_base_path


def interrupt_liveview(f):
    def wrapper(self):
        log.debug('stopping liveview')
        self.timer.stop()
        log.debug('waiting for image grabing thread to finish')
        # wait 2*exposure time for detector to finish; otherwise abort
        if self.dectris_image_grabber.connected:
            for _ in range(int(self.dectris_image_grabber.Q.frame_time)):
                if self.dectris_image_grabber.image_grabber_thread.isFinished():
                    break
                sleep(0.002)
            if not self.dectris_image_grabber.image_grabber_thread.isFinished():
                log.warning('image grabbing thread does not seem to finish, aborting acquisition')
                if self.dectris_image_grabber.connected:
                    self.dectris_image_grabber.Q.abort()
        f(self)
        log.debug('restarting liveview')
        self.timer.start(self.update_interval)
    return wrapper


class LiveViewUi(QtWidgets.QMainWindow):
    image = None
    i_digits = None
    update_interval = None

    def __init__(self, cmd_args, *args, **kwargs):
        log.debug('initializing DectrisLiveView')
        super().__init__(*args, **kwargs)
        uic.loadUi(path.join(get_base_path(), 'ui/liveview.ui'), self)

        self.comboBoxTriggerMode.currentIndexChanged.connect(self.update_trigger_mode)
        self.spinBoxExposure.valueChanged.connect(self.update_exposure)

        self.viewer.cursor_changed.connect(self.update_statusbar)

        self.dectris_image_grabber = DectrisImageGrabber(cmd_args.ip, cmd_args.port)
        self.dectris_image_grabber.image_ready.connect(self.update_image)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.update_interval = cmd_args.update_interval
        self.timer.start(self.update_interval)

        self.show()

    def update_statusbar(self, xy):
        log.debug(f'updating statusbar with xy: {xy}')
        if self.image is None:
            self.statusbar.showMessage('')
            return
        if xy == (np.NaN, np.NaN):  # triggered when cursor outside of image
            self.statusbar.showMessage('')
            return
        x, y = xy
        i = self.image[x, y]
        self.statusbar.showMessage(f'({x:>4}, {y:>4}) | I={i:{self.i_digits}.0f}')

    def update_image(self, image):
        self.image = image
        self.viewer.x_size, self.viewer.y_size = self.image.shape
        self.viewer.setImage(self.image, autoRange=False, autoLevels=False)
        self.i_digits = len(str(int(self.image.max(initial=1))))
        self.statusbar.showMessage('')

    def update_ui(self):
        self.dectris_image_grabber.image_grabber_thread.start()

    @interrupt_liveview
    def update_trigger_mode(self):
        mode = self.comboBoxTriggerMode.currentText()
        if mode == 'exts':
            self.spinBoxExposure.setEnabled(False)
        else:
            self.spinBoxExposure.setEnabled(True)
        log.info(f'changing trigger mode to {mode}')
        if self.dectris_image_grabber.connected:
            self.dectris_image_grabber.Q.trigger_mode = mode
        else:
            log.warning(f'could not change trigger mode, detector disconnected')

    @interrupt_liveview
    def update_exposure(self):
        time = self.spinBoxExposure.value()
        log.info(f'changing exporue time to {time}')
        if self.dectris_image_grabber.connected:
            self.dectris_image_grabber.Q.count_time = time
            self.dectris_image_grabber.Q.frame_time = time
        else:
            log.warning(f'could not change exposure time, detector disconnected')
