from os import path
import logging as log
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, uic
import pyqtgraph as pg
from .. import get_base_path
from ..lib.Utils import DectrisImageGrabber, DectrisStatusGrabber, ConstantPing, interrupt_acquisition
from .widgets import ROIView
from ..ui.captured import CapturedUi


class LiveViewUi(QtWidgets.QMainWindow):
    """
    main window of the LiveView application
    """
    image = None
    x_projection = pg.PlotCurveItem()
    y_projection = pg.PlotCurveItem()
    i_digits = 5
    update_interval = None

    def __init__(self, cmd_args, *args, **kwargs):
        log.debug('initializing DectrisLiveView')
        super().__init__(*args, **kwargs)
        uic.loadUi(path.join(get_base_path(), 'ui/liveview.ui'), self)
        self.update_interval = cmd_args.update_interval

        self.dectris_image_grabber = DectrisImageGrabber(cmd_args.ip, cmd_args.port,
                                                         trigger_mode='ints',
                                                         exposure=float(self.lineEditExposure.text()) / 1000)
        self.dectris_status_grabber = DectrisStatusGrabber(cmd_args.ip, cmd_args.port)
        self.exposure_progress_worker = ConstantPing()
        self.dectris_image_grabber.exposure_triggered.connect(self.exposure_progress_worker.progress_thread.start)

        self.image_timer = QtCore.QTimer()
        self.image_timer.timeout.connect(self.dectris_image_grabber.image_grabber_thread.start)
        self.dectris_image_grabber.image_ready.connect(self.update_image)

        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.dectris_status_grabber.status_grabber_thread.start)
        self.dectris_status_grabber.status_ready.connect(self.update_status_labels)

        self.exposure_progress_worker.advance_progress_bar.connect(self.advance_progress_bar)

        self.lineEditExposure.returnPressed.connect(self.update_exposure)
        self.lineEditCapture.returnPressed.connect(self.capture_image)

        self.labelIntensity = QtWidgets.QLabel()
        self.labelState = QtWidgets.QLabel()
        self.labelTrigger = QtWidgets.QLabel()
        self.labelExposure = QtWidgets.QLabel()

        self.init_menubar()
        self.init_statusbar()
        self.reset_progress_bar()

        self.status_timer.start(200)

        self.roi_view = ROIView(title='ROIs')

        self.viewer.addItem(self.x_projection)
        self.viewer.addItem(self.y_projection)
        self.show()

    def closeEvent(self, evt):
        self.roi_view.hide()
        self.hide()
        self.image_timer.stop()
        self.status_timer.stop()
        self.dectris_image_grabber.image_grabber_thread.requestInterruption()
        self.exposure_progress_worker.progress_thread.requestInterruption()
        self.exposure_progress_worker.progress_thread.wait()
        self.dectris_status_grabber.status_grabber_thread.wait()
        self.dectris_image_grabber.image_grabber_thread.wait()
        super().closeEvent(evt)

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

    def init_menubar(self):
        self.actionAddRectangle.triggered.connect(self.add_rect_roi)
        self.actionAddRectangle.setShortcut('R')
        self.actionRemoveLastROI.triggered.connect(self.remove_last_roi)
        self.actionRemoveLastROI.setShortcut('Shift+R')
        self.actionRemoveAllROIs.triggered.connect(self.remove_all_rois)
        self.actionRemoveAllROIs.setShortcut('Ctrl+Shift+R')
        self.actionLinkYAxis.triggered.connect(self.update_y_axis_link)
        self.actionLinkYAxis.setShortcut('Y')
        self.actionAutoRange.setShortcut('A')
        self.actionShowProjections.setShortcut('P')
        self.actionShowMaxPixelValue.triggered.connect(self.update_show_max_pixel_value)
        self.actionShowMaxPixelValue.setShortcut('M')
        self.actionShowFrame.triggered.connect(self.update_show_frame)
        self.actionShowFrame.setShortcut('F')

        trigger_mode_group = QtWidgets.QActionGroup(self)
        trigger_mode_group.addAction(self.actionINTS)
        trigger_mode_group.addAction(self.actionEXTS)
        trigger_mode_group.addAction(self.actionEXTE)
        trigger_mode_group.addAction(self.actionStop)
        self.actionINTS.triggered.connect(self.update_trigger_mode)
        self.actionINTS.setShortcut('Ctrl+1')
        self.actionEXTS.triggered.connect(self.update_trigger_mode)
        self.actionEXTS.setShortcut('Ctrl+2')
        self.actionEXTE.triggered.connect(self.update_trigger_mode)
        self.actionEXTE.setShortcut('Ctrl+3')
        self.actionStop.triggered.connect(self.update_trigger_mode)
        self.actionStop.setShortcut('Esc')

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
            self.labelExposure.setText(f'Exposure: {states["exposure"] * 1000:>5.0f}ms')

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, image):
        self.image = image
        self.viewer.clear()
        self.viewer.setImage(image, autoRange=self.actionAutoRange.isChecked(),
                             autoLevels=self.actionAutoRange.isChecked())
        if self.actionShowProjections.isChecked():
            x_projection_data = np.mean(image, axis=0)
            x_projection_data /= np.mean(x_projection_data)
            x_projection_data *= image.shape[1]*0.1
            self.x_projection.setData(x=x_projection_data, y=np.arange(0, image.shape[1])+0.5)

            y_projection_data = np.mean(image, axis=1)
            y_projection_data /= np.max(y_projection_data)
            y_projection_data *= image.shape[0]*0.1  # make plot span 10% of the image
            self.y_projection.setData(x=np.arange(0, image.shape[0])+0.5, y=y_projection_data)
        else:
            self.x_projection.clear()
            self.y_projection.clear()
        self.i_digits = len(str(int(image.max(initial=1))))
        self.update_all_rois()
        self.exposure_progress_worker.progress_thread.requestInterruption()
        self.exposure_progress_worker.progress_thread.wait()
        self.reset_progress_bar()

    @interrupt_acquisition
    @QtCore.pyqtSlot()
    def capture_image(self):
        log.info('capturing image')
        if self.dectris_image_grabber.connected:
            if self.dectris_image_grabber.Q.trigger_mode == 'ints':
                try:
                    time = float(self.lineEditCapture.text()) / 1000
                except (ValueError, TypeError):
                    log.warning(f'image capture: cannot convert {self.lineEditCapture.text()} to float')
                    return
                
                self.dectris_image_grabber.Q.trigger_mode = 'ints'
                self.dectris_image_grabber.Q.count_time = time
                self.dectris_image_grabber.Q.frame_time = time

                self.dectris_image_grabber.image_ready.disconnect(self.update_image)
                self.dectris_image_grabber.image_ready.connect(self.show_captured_image)
                self.dectris_image_grabber.image_grabber_thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def show_captured_image(self, image):
        log.info('showing captured image')
        self.dectris_image_grabber.image_ready.disconnect(self.show_captured_image)
        self.dectris_image_grabber.image_ready.connect(self.update_image)
        CapturedUi(image, parent=self)
        self.update_exposure()
        self.update_trigger_mode()

    @interrupt_acquisition
    @QtCore.pyqtSlot()
    def update_trigger_mode(self):
        if self.actionStop.isChecked():
            self.image_timer.stop()
            self.progressBarExposure.setValue(self.progressBarExposure.minimum())
        else:
            if not self.image_timer.isActive():
                self.image_timer.start(self.update_interval)
            if self.actionINTS.isChecked():
                mode = 'ints'
                self.lineEditExposure.setEnabled(True)
            elif self.actionEXTS.isChecked():
                mode = 'exts'
                self.lineEditExposure.setEnabled(True)
            else:
                mode = 'exte'
                self.lineEditExposure.setEnabled(False)
            log.info(f'changing trigger mode to {mode}')
            if self.dectris_image_grabber.connected:
                self.dectris_image_grabber.Q.trigger_mode = mode
            else:
                log.warning(f'could not change trigger mode, detector disconnected')

    @interrupt_acquisition
    @QtCore.pyqtSlot()
    def update_exposure(self):
        try:
            time = float(self.lineEditExposure.text()) / 1000
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
        if self.progressBarExposure.value() + 1 < self.progressBarExposure.maximum():
            self.progressBarExposure.setValue(self.progressBarExposure.value() + 1)
        else:
            self.progressBarExposure.setValue(self.progressBarExposure.maximum())

    def reset_progress_bar(self):
        if self.dectris_image_grabber.connected:
            time = self.progressBarExposure.setMaximum(int(self.dectris_image_grabber.Q.frame_time * 100))
        else:
            time = 100
        self.progressBarExposure.setValue(0)
        if time is not None:
            self.progressBarExposure.setMaximum(time)

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        self.dectris_image_grabber.image_grabber_thread.start()

    @QtCore.pyqtSlot()
    def add_rect_roi(self):
        if self.image is not None:
            log.info('added rectangular ROI')
            roi = pg.RectROI((self.image.shape[0] / 2 - 50, self.image.shape[1] / 2 - 50), (100, 100),
                             centered=True, sideScalers=True,
                             pen=pg.mkPen('c', width=2), hoverPen=pg.mkPen('c', width=3),
                             handlePen=pg.mkPen('w', width=3), handleHoverPen=pg.mkPen('w', width=4))
            roi.removable = True
            roi.sigRemoveRequested.connect(self.remove_roi)
            roi.sigRegionChanged.connect(self.update_roi)

            self.viewer.addItem(roi)
            roi.plot_item = self.roi_view.addPlot()
            roi.plot_item.setMouseEnabled(x=False, y=True)
            if len(self.roi_view) > 1 and self.actionLinkYAxis.isChecked():
                roi.plot_item.setYLink(self.roi_view.plots[0])

            self.roi_view.rearrange()
            self.update_roi(roi)
            self.roi_view.show()

        else:
            log.warning('cannot add ROI before an image is dislayed')

    @QtCore.pyqtSlot(tuple)
    def update_roi(self, roi):
        roi_data = roi.getArrayRegion(self.image, self.viewer.imageItem)
        roi.plot_item.clear()
        roi.plot_item.plot(roi_data.mean(axis=np.argmin(roi_data.shape)))

    def update_all_rois(self):
        for i in self.viewer.view.addedItems:
            if isinstance(i, pg.RectROI):
                try:
                    self.update_roi(i)
                except Exception:  # bad practice, but works for now...
                    pass

    @QtCore.pyqtSlot(tuple)
    def remove_roi(self, roi):
        self.viewer.scene.removeItem(roi)
        self.roi_view.removeItem(roi.plot_item)
        if len(self.roi_view) == 0:
            self.roi_view.hide()

    @QtCore.pyqtSlot()
    def update_y_axis_link(self):
        self.roi_view.set_link_y_axis(self.actionLinkYAxis.isChecked())

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        self.dectris_image_grabber.image_grabber_thread.start()

    @QtCore.pyqtSlot()
    def remove_last_roi(self):
        for i in self.viewer.view.addedItems[::-1]:
            if isinstance(i, pg.ROI):
                try:
                    self.remove_roi(i)
                    return
                except Exception:  # again bad practice, but works...
                    pass

    @QtCore.pyqtSlot()
    def remove_all_rois(self):
        for i in self.viewer.view.addedItems[::-1]:
            if isinstance(i, pg.ROI):
                try:
                    self.remove_roi(i)
                except Exception:  # again bad practice, but works...
                    pass

    @QtCore.pyqtSlot()
    def update_show_max_pixel_value(self):
        self.viewer.show_max = self.actionShowMaxPixelValue.isChecked()

    @QtCore.pyqtSlot()
    def update_show_frame(self):
        self.viewer.show_frame = self.actionShowFrame.isChecked()
