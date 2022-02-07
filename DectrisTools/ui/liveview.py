from os import path
import logging as log
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from ..lib.Utils import DectrisImageGrabber, DectrisStatusGrabber, ExposureProgressWorker, interrupt_liveview
from .. import get_base_path


class LiveViewUi(QtWidgets.QMainWindow):
    image = None
    i_digits = 5
    update_interval = None

    def __init__(self, cmd_args, *args, **kwargs):
        log.debug('initializing DectrisLiveView')
        super().__init__(*args, **kwargs)
        uic.loadUi(path.join(get_base_path(), 'ui/liveview.ui'), self)
        self.update_interval = cmd_args.update_interval

        self.dectris_image_grabber = DectrisImageGrabber(cmd_args.ip, cmd_args.port,
                                                         trigger_mode=self.comboBoxTriggerMode.currentText(),
                                                         exposure=float(self.lineEditExposure.text())/1000)
        self.dectris_status_grabber = DectrisStatusGrabber(cmd_args.ip, cmd_args.port)
        self.exposure_progress_worker = ExposureProgressWorker()
        self.dectris_image_grabber.exposure_triggered.connect(self.exposure_progress_worker.progress_thread.start)

        self.image_timer = QtCore.QTimer()
        self.image_timer.timeout.connect(self.dectris_image_grabber.image_grabber_thread.start)
        self.dectris_image_grabber.image_ready.connect(self.update_image)

        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.dectris_status_grabber.status_grabber_thread.start)
        self.dectris_status_grabber.status_ready.connect(self.update_status_labels)

        self.exposure_progress_worker.advance_progress_bar.connect(self.advance_progress_bar)

        self.labelIntensity = QtWidgets.QLabel()
        self.labelState = QtWidgets.QLabel()
        self.labelTrigger = QtWidgets.QLabel()
        self.labelExposure = QtWidgets.QLabel()

        self.comboBoxTriggerMode.currentIndexChanged.connect(self.update_trigger_mode)
        self.lineEditExposure.returnPressed.connect(self.update_exposure)

        self.init_statusbar()
        self.reset_progress_bar()

        self.image_timer.start(self.update_interval)
        self.status_timer.start(200)

        self.show()

    def init_statusbar(self):
        self.viewer.cursor_changed.connect(self.update_label_intensity)

        status_label_font = QtGui.QFont('Courier', 9)
        self.labelIntensity.setFont(status_label_font)
        self.labelState.setFont(status_label_font)
        self.labelTrigger.setFont(status_label_font)
        self.labelExposure.setFont(status_label_font)

        self.labelIntensity.setText(f'({"":>4s}, {"":>4s})   {"":>{self.i_digits}s}')

        self.statusbar.addPermanentWidget(self.labelIntensity)
        self.statusbar.addPermanentWidget(self.labelState)
        self.statusbar.addPermanentWidget(self.labelTrigger)
        self.statusbar.addPermanentWidget(self.labelExposure)

    @QtCore.pyqtSlot(tuple)
    def update_label_intensity(self, xy):
        if self.image is None or xy == (np.NaN, np.NaN):
            self.labelIntensity.setText(f'({"":>4s}, {"":>4s})   {"":>{self.i_digits}s}')
            return
        x, y = xy
        i = self.image[x, y]
        self.labelIntensity.setText(f'({x:>4}, {y:>4}) I={i:>{self.i_digits}.0f}')

    @QtCore.pyqtSlot(dict)
    def update_status_labels(self, states):
        if states['quadro'] is None:
            self.labelState.setText(f'Detector: {"":>7s} Monitor: {"":>7s}')
            self.labelTrigger.setText(f'Trigger: {"":>4s}')
            self.labelExposure.setText(f'Exposure: {"":>5s}  ')
        else:
            self.labelState.setText(f'Detector: {states["quadro"]:>7s} Monitor: {states["mon"]:>7s}')
            self.labelTrigger.setText(f'Trigger: {states["trigger_mode"]:>4s}')
            if states['trigger_mode'] == 'exts':
                self.labelExposure.setText('Exposure:   trig ')
            self.labelExposure.setText(f'Exposure: {states["exposure"]*1000:>5.0f}ms')

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, image):
        self.image = image
        self.viewer.x_size, self.viewer.y_size = self.image.shape
        self.viewer.setImage(self.image, autoRange=False, autoLevels=False)
        self.i_digits = len(str(int(self.image.max(initial=1))))
        self.statusbar.showMessage('')
        self.exposure_progress_worker.progress_thread.quit()
        self.reset_progress_bar()

    @interrupt_liveview
    @QtCore.pyqtSlot()
    def update_trigger_mode(self):
        mode = self.comboBoxTriggerMode.currentText()
        if mode == 'exts':
            self.lineEditExposure.setEnabled(False)
        else:
            self.lineEditExposure.setEnabled(True)
        log.info(f'changing trigger mode to {mode}')
        if self.dectris_image_grabber.connected:
            self.dectris_image_grabber.Q.trigger_mode = mode
        else:
            log.warning(f'could not change trigger mode, detector disconnected')

    @interrupt_liveview
    @QtCore.pyqtSlot()
    def update_exposure(self):
        try:
            time = float(self.lineEditExposure.text())/1000
        except (ValueError, TypeError):
            log.warning(f'setting exposure: cannot convert {self.lineEditExposure.text()} to float')
            return

        log.info(f'changing exporue time to {time}')
        if self.dectris_image_grabber.connected:
            self.dectris_image_grabber.Q.count_time = time
            self.dectris_image_grabber.Q.frame_time = time
        else:
            log.warning(f'could not change exposure time, detector disconnected')

    @QtCore.pyqtSlot()
    def advance_progress_bar(self):
        if self.progressBarExposure.value()+1 < self.progressBarExposure.maximum():
            self.progressBarExposure.setValue(self.progressBarExposure.value()+1)

    def reset_progress_bar(self):
        if self.dectris_image_grabber.connected:
            time = self.progressBarExposure.setMaximum(int(self.dectris_image_grabber.Q.frame_time*100))
        else:
            time = 500
        self.progressBarExposure.setValue(0)
        self.progressBarExposure.setMaximum(time)

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        self.dectris_image_grabber.image_grabber_thread.start()
