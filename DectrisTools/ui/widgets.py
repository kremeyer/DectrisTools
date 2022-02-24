from copy import copy, deepcopy
import logging as log
import pyqtgraph as pg
from numpy import sqrt, ceil, NaN
from PyQt5.QtCore import pyqtSignal, pyqtSlot


class ImageViewWidget(pg.ImageView):
    x_size = 0
    y_size = 0
    cursor_changed = pyqtSignal(tuple)

    def __init__(self, parent=None, show_max=True, show_frame=False, cmap='inferno'):
        log.debug('initializing ImageViewWidget')
        super().__init__()
        self.setParent(parent)
        self.setPredefinedGradient(cmap)
        self.setLevels(0, 2**16)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        self.max_label = pg.LabelItem(justify='right')
        self.addItem(self.max_label)

        self.frame_top = pg.InfiniteLine(angle=0, movable=False)
        self.frame_bottom = pg.InfiniteLine(angle=0, movable=False)
        self.frame_left = pg.InfiniteLine(angle=90, movable=False)
        self.frame_right = pg.InfiniteLine(angle=90, movable=False)
        self.frames = [self.frame_top, self.frame_bottom, self.frame_left, self.frame_right]

        self.view.invertY(True)
        self.proxy = pg.SignalProxy(self.scene.sigMouseMoved,
                                    rateLimit=60, slot=self.__callback_move)
        self.show_max = show_max
        self.show_frame = show_frame

    def setImage(self, *args, **kwargs):
        self.x_size, self.y_size = args[0].shape
        if self.show_max:
            self.max_label.setText(f'<span style="font-size: 32pt">{int(args[0].max())}</span>')
        else:
            self.max_label.setText('')
        if self.show_frame:
            self.frame_top.setPos(self.y_size)
            self.frame_bottom.setPos(0)
            self.frame_left.setPos(0)
            self.frame_right.setPos(self.x_size)
            for f in self.frames:
                self.addItem(f)
        else:
            items_in_view = copy(self.view.addedItems)
            for i in items_in_view:
                if isinstance(i, pg.InfiniteLine):
                    try:
                        self.view.addedItems.remove(i)
                        self.view.removeItem(i)
                    except ValueError:
                        pass
        super().setImage(*args, **kwargs)

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
