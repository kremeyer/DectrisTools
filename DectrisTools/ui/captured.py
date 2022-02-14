from os import path
import logging as log
from datetime import datetime
import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from .. import get_base_path


class CapturedUi(QtWidgets.QMainWindow):
    """
    window for displaying captured images
    """
    def __init__(self, image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(path.join(get_base_path(), 'ui/captured.ui'), self)
        self.viewer.setImage(image, autoRange=True, autoLevels=True)
        self.image = image
        self.i_digits = len(str(int(self.image.max(initial=1))))
        self.viewer.x_size, self.viewer.y_size = self.image.shape

        self.viewer.cursor_changed.connect(self.update_statusbar)
        self.setWindowTitle(f'Captured Image - {datetime.now().strftime("%Y%m%d %H%M%S")}')
        self.show()

    @QtCore.pyqtSlot(tuple)
    def update_statusbar(self, xy):
        if xy == (np.NaN, np.NaN):
            self.statusbar.showMessage('')
            return
        x, y = xy
        i = self.image[x, y]
        self.statusbar.showMessage(f'({x:4d}, {y:4d}) | I={i:{self.i_digits}.0f}')
