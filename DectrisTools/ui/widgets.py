from copy import copy, deepcopy
import logging as log
import weakref
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ui_template
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot


class ImageViewWidget(pg.ImageView):
    x_size = 0
    y_size = 0
    image = None
    raw_image = None
    cursor_changed = pyqtSignal(tuple)

    def __init__(self, parent=None, cmap='inferno'):
        log.debug('initializing ImageViewWidget')
        super().__init__()
        self.setParent(parent)
        self.setPredefinedGradient(cmap)
        self.setLevels(0, 2**16)
        self.ui.histogram.axis.setRange(0, 2**16)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.view.invertY(True)
        self.view.setAspectLocked(1)

        self.view.menu = ViewBoxMenu(self.view)
        self.view.menu.autoLevels.triggered.connect(self.update_scale)
        self.view.menu.linScale.triggered.connect(self.update_scale)
        self.view.menu.logScale.triggered.connect(self.update_scale)
        self.view.menu.sqrtScale.triggered.connect(self.update_scale)

        self.proxy = pg.SignalProxy(self.scene.sigMouseMoved,
                                    rateLimit=60, slot=self.__callback_move)

        self.max_label = pg.LabelItem(justify='right')
        self.frame_top = pg.InfiniteLine(angle=0, movable=False)
        self.frame_bottom = pg.InfiniteLine(angle=0, movable=False)
        self.frame_left = pg.InfiniteLine(angle=90, movable=False)
        self.frame_right = pg.InfiniteLine(angle=90, movable=False)
        self.frames = [self.frame_top, self.frame_bottom, self.frame_left, self.frame_right]

        self.crosshair_h = pg.InfiniteLine(angle=45, movable=False)
        self.crosshair_v = pg.InfiniteLine(angle=135, movable=False)

        self.x_projection = pg.PlotCurveItem()
        self.addItem(self.x_projection)
        self.y_projection = pg.PlotCurveItem()
        self.addItem(self.y_projection)

        self.addItem(self.max_label)

    def setImage(self, *args, max_label=False, projections=False, **kwargs):
        self.raw_image = copy(args[0])
        self.image = args[0]
        self.x_size, self.y_size = self.image.shape

        if self.view.menu.logScale.isChecked():
            self.image = np.log(self.image, where=self.image > 0)
        elif self.view.menu.sqrtScale.isChecked():
            self.image = np.sqrt(self.image, where=self.image > 0)

        if max_label:
            self.max_label.setText(f'<span style="font-size: 32pt">{int(self.image.max())}</span>')
        else:
            self.max_label.setText('')

        if projections:
            x_projection_data = np.mean(self.image, axis=0)
            x_projection_data /= np.mean(x_projection_data)
            x_projection_data *= self.image.shape[1] * 0.1
            self.x_projection.setData(x=x_projection_data, y=np.arange(0, self.image.shape[1]) + 0.5)

            y_projection_data = np.mean(self.image, axis=1)
            y_projection_data /= np.max(y_projection_data)
            y_projection_data *= self.image.shape[0] * 0.1  # make plot span 10% of the image
            self.y_projection.setData(x=np.arange(0, self.image.shape[0]) + 0.5, y=y_projection_data)
        else:
            self.x_projection.clear()
            self.y_projection.clear()

        auto_levels = self.view.menu.autoLevels.isChecked()
        super().setImage(self.image, *args[1:], autoLevels=auto_levels,
                         autoHistogramRange=auto_levels, autoRange=False, **kwargs)

    @pyqtSlot()
    def update_scale(self, *args, **kwargs):
        if self.raw_image is None:
            return
        if self.view.menu.logScale.isChecked():
            self.raw_image = np.log(self.raw_image, where=self.raw_image > 0)
        elif self.view.menu.sqrtScale.isChecked():
            self.raw_image = np.sqrt(self.raw_image, where=self.raw_image > 0)
        auto_levels = self.view.menu.autoLevels.isChecked()
        super().setImage(self.raw_image, *args, autoLevels=auto_levels,
                         autoHistogramRange=auto_levels, autoRange=False, **kwargs)

    @pyqtSlot(tuple)
    def __callback_move(self, evt):
        """
        callback function for mouse movement on image
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

    @pyqtSlot(bool)
    def show_frame(self, state):
        if state:
            self.frame_top.setPos(self.y_size)
            self.frame_bottom.setPos(0)
            self.frame_left.setPos(0)
            self.frame_right.setPos(self.x_size)
            for f in self.frames:
                self.addItem(f)
            return
        items_in_view = copy(self.view.addedItems)
        for i in items_in_view:
            if isinstance(i, pg.InfiniteLine):
                if i.angle in [0, 90]:
                    try:
                        self.view.addedItems.remove(i)
                        self.view.removeItem(i)
                    except ValueError:
                        pass

    @pyqtSlot(bool)
    def show_crosshair(self, state):
        if state:
            self.crosshair_h.setPos((self.x_size/2, self.y_size/2))
            self.addItem(self.crosshair_h)
            self.crosshair_v.setPos((self.x_size/2, self.y_size/2))
            self.addItem(self.crosshair_v)
            return
        items_in_view = copy(self.view.addedItems)
        for i in items_in_view:
            if isinstance(i, pg.InfiniteLine):
                if i.angle in [45, 135]:
                    try:
                        self.view.addedItems.remove(i)
                        self.view.removeItem(i)
                    except ValueError:
                        pass


class ViewBoxMenu(pg.graphicsItems.ViewBox.ViewBoxMenu.ViewBoxMenu):
    """
    this is derived from pg.graphicsItems.ViewBox.ViewBoxMenu.ViewBoxMenu in order to adapt the context menu
    """
    def __init__(self, view):
        QtWidgets.QMenu.__init__(self)

        self.view = weakref.ref(view)  # keep weakref to view to avoid circular reference
        self.valid = False  # tells us whether the ui needs to be updated
        self.viewMap = weakref.WeakValueDictionary()  # weakrefs to all views listed in the link combos

        self.setTitle('ViewBox options')
        self.viewAll = QtGui.QAction('View All', self)
        self.viewAll.triggered.connect(self.autoRange)
        self.addAction(self.viewAll)

        self.autoLevels = QtGui.QAction('Auto Levels', self)
        self.autoLevels.setCheckable(True)
        self.autoLevels.setChecked(True)
        self.addAction(self.autoLevels)

        self.addSeparator()
        g = QtWidgets.QActionGroup(self)
        self.linScale = QtGui.QAction('Linear Scale')
        self.linScale.setCheckable(True)
        self.linScale.setChecked(True)
        self.addAction(self.linScale)
        g.addAction(self.linScale)
        self.logScale = QtGui.QAction('Log Scale')
        self.logScale.setCheckable(True)
        self.addAction(self.logScale)
        g.addAction(self.logScale)
        self.sqrtScale = QtGui.QAction('Sqrt Scale')
        self.sqrtScale.setCheckable(True)
        self.addAction(self.sqrtScale)
        g.addAction(self.sqrtScale)

        # unused actions
        self.axes = []
        self.ctrl = []
        self.widgetGroups = []
        self.dv = QtGui.QDoubleValidator(self)
        for _ in 'XY':
            w = QtWidgets.QWidget()
            ui = ui_template.Ui_Form()
            ui.setupUi(w)
            a = QtWidgets.QWidgetAction(self)
            a.setDefaultWidget(w)
            self.ctrl.append(ui)
        self.leftMenu = QtWidgets.QMenu('Mouse Mode')
        pan = QtGui.QAction('3 button', self.leftMenu)
        zoom = QtGui.QAction('1 button', self.leftMenu)
        self.mouseModes = [pan, zoom]
        self.view().sigStateChanged.connect(self.viewStateChanged)
        self.updateState()


class ROIView(pg.GraphicsLayoutWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackground('k')

    def __len__(self):
        return len(self.ci.items)

    @property
    def plots(self):
        try:
            return list(deepcopy(self.ci.items).keys())
        except TypeError:
            return list(copy(self.ci.items).keys())

    def rearrange(self):
        log.info('rearranging ROIView layout')
        rows = round(np.sqrt(len(self)))
        cols = int(np.ceil(len(self)/rows))
        plots = self.plots
        self.clear()
        i = 0
        for r in range(rows):
            for c in range(cols):
                if i < len(plots):
                    self.ci.addItem(plots[i], r, c)
                    i += 1

    def set_link_y_axis(self, state):
        log.info(f'setting link y axis to {state}')
        if state:
            for p in self.plots[1:]:
                p.setYLink(self.plots[0])
        else:
            for p in self.plots:
                p.setYLink(None)
