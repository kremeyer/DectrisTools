from os import path
import logging as log
import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from ..lib.Utils import DectrisGrabber
from .. import get_base_path


class LiveViewUi(QtWidgets.QMainWindow):
    image = None
    i_digits = None

    def __init__(self, cmd_args, *args, **kwargs):
        log.debug('initializing DectrisLiveView')
        super().__init__(*args, **kwargs)
        uic.loadUi(path.join(get_base_path(), 'ui/liveview.ui'), self)

        self.viewer.cursor_changed.connect(self.update_statusbar)

        self.dectris_grabber = DectrisGrabber(cmd_args.ip, cmd_args.port)
        self.dectris_grabber.image_ready.connect(self.update_image)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(cmd_args.update_interval)

        self.show()

    def update_statusbar(self, xy):
        log.debug(f'updating statusbar with xy: {xy}')
        if self.image is None:
            self.statusbar.showMessage('')
            return
        if xy == (np.NaN, np.NaN):  # triggered when cursor outside the image
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
        self.dectris_grabber.image_grabber_thread.start()
