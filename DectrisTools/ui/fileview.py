import logging as log
from PyQt5 import QtWidgets, QtCore


class FileViewUi(QtWidgets.QMainWindow):
    """
    main window of the fileview application
    """
    def __init__(self, *args, **kwargs):
        log.debug("initializing DectrisFileView")
        super().__init__(*args, **kwargs)

        self.settings = QtCore.QSettings(
            "Siwick Research Group", "DectrisTools Fileview", parent=self
        )
        if self.settings.value("main_window_geometry") is not None:
            self.setGeometry(self.settings.value("main_window_geometry"))

        self.show()

    def closeEvent(self, evt):
        self.settings.setValue("main_window_geometry", self.geometry())
        super().closeEvent(evt)
