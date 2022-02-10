from copy import copy, deepcopy
import logging as log
import pyqtgraph as pg
from numpy import sqrt, ceil, NaN
from PyQt5.QtCore import pyqtSignal, pyqtSlot


class LiveViewWidget(pg.ImageView):
    x_size = 0
    y_size = 0
    cursor_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        log.debug('initializing LiveView')
        super().__init__()
        self.setParent(parent)
        self.setPredefinedGradient('inferno')
        self.setLevels(0, 2**16)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.view.invertY(False)
        self.proxy = pg.SignalProxy(self.scene.sigMouseMoved,
                                    rateLimit=60, slot=self.__callback_move)

    @pyqtSlot(tuple)
    def __callback_move(self, evt):
        """
        callback function for mouse movement on image
        """
        qpoint = self.view.mapSceneToView(evt[0])
        x = int(qpoint.x())
        y = int(qpoint.y())
        if x < 0 or x >= self.x_size:
            self.cursor_changed.emit((NaN, NaN))
            return
        if y < 0 or y >= self.y_size:
            self.cursor_changed.emit((NaN, NaN))
            return
        self.cursor_changed.emit((x, y))


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
        rows = round(sqrt(len(self)))
        cols = int(ceil(len(self)/rows))
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
