import os
from collections.abc import Iterable
from os import path
import logging as log
import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from PIL import Image
from .. import get_base_path


class FileViewUi(QtWidgets.QMainWindow):
    """
    main window of the fileview application
    """
    __PIL_IMAGE_FORMATS = ('.npy', '.tif', '.tiff', '.bmp', '.eps', '.gif', '.jpeg', '.jpg', '.png')

    def __init__(self, *args, **kwargs):
        log.debug("initializing DectrisFileView")
        super().__init__(*args, **kwargs)
        uic.loadUi(path.join(get_base_path(), "ui/fileview.ui"), self)
        self.setAcceptDrops(True)

        self.settings = QtCore.QSettings(
            "Siwick Research Group", "DectrisTools Fileview", parent=self
        )
        if self.settings.value("main_window_geometry") is not None:
            self.setGeometry(self.settings.value("main_window_geometry"))
        if self.settings.value("pin_histogram_zero") is not None:
            pin_zero = self.settings.value("pin_histogram_zero").lower() == "true"
            self.actionPinHistogramZero.setChecked(pin_zero)
        if self.settings.value("image_levels") is not None:
            self.viewer.setLevels(*self.settings.value("image_levels"))
            self.viewer.setHistogramRange(*self.settings.value("image_levels"))
        if self.settings.value("histogram_range") is not None:
            self.viewer.ui.histogram.setHistogramRange(
                *self.settings.value("histogram_range"), padding=0
            )

        self.init_menubar()

        self.show()

    def closeEvent(self, evt):
        self.settings.setValue("main_window_geometry", self.geometry())
        self.settings.setValue("image_levels", self.viewer.getLevels())
        self.settings.setValue(
            "pin_histogram_zero", self.actionPinHistogramZero.isChecked()
        )
        # this is now easier, can be changed, when new version of pyqtgraph is released
        # https://github.com/pyqtgraph/pyqtgraph/pull/2397
        hist_range = tuple(self.viewer.ui.histogram.item.vb.viewRange()[1])  # wtf?
        self.settings.setValue("histogram_range", hist_range)
        super().closeEvent(evt)

    def init_menubar(self):
        self.actionShowCrosshair.setShortcut("C")
        self.actionShowFrame.setShortcut("F")
        self.actionPinHistogramZero.setShortcut("H")
        self.actionShowCrosshair.triggered.connect(
            lambda x=self.actionShowCrosshair.isChecked(): self.viewer.show_crosshair(x)
        )
        self.actionShowFrame.triggered.connect(
            lambda x=self.actionShowFrame.isChecked(): self.viewer.show_frame(x)
        )
        self.actionPinHistogramZero.triggered.connect(self.set_pin_histogram_zero)

    def set_pin_histogram_zero(self):
        self.viewer.pin_histogram_zero = self.actionPinHistogramZero.isChecked()

    def dragEnterEvent(self, event):
        """
        checks if the event contains an url
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Checks what has been dropped onto the application and takes appropriate action. Can load one or more image
        files. If directories are dropped, all files in them are loaded (no recursion).
        """
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        images = []
        for p in paths:
            if path.isfile(p):
                from_file = self.load_from_file(p)
                if isinstance(from_file, Iterable):
                    for image in from_file:
                        images.append(image)
                else:
                    images.append(from_file)
            elif path.isdir(p):
                for file in os.listdir(p):
                    if path.isfile(file):
                        from_file = self.load_from_file(p)
                        if isinstance(from_file, Iterable):
                            for image in from_file:
                                images.append(image)
                        else:
                            images.append(from_file)
        # discard images that don't match size of the first file
        images = [i for i in images if i.shape == images[0].shape]
        if images:
            self.viewer.setImage(np.array(images))

    def load_from_file(self, file):
        images = []
        if path.isfile(file):
            if file.lower().endswith(self.__PIL_IMAGE_FORMATS):
                image_array = np.array(Image.open(file))
                if image_array.ndim == 3:
                    image_array = np.mean(image_array, axis=2)
                images.append(image_array)
            if file.lower().endswith('.h5'):
                pass
        return images
