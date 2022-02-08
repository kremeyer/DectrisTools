from copy import copy, deepcopy
import pyqtgraph as pg
from numpy import sqrt, ceil


class ROIView(pg.GraphicsLayoutWidget):
    rois = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.ci.items)

    def add_roi(self, roi):
        self.rois.append(roi)

    def rearrange(self):
        rows = int(round(sqrt(len(self))))
        cols = int(ceil(rows/round(sqrt(len(self)))))
        try:
            plots = deepcopy(self.get_plots())
        except TypeError:
            plots = copy(self.get_plots())
        self.clear()
        for plot, row, col in zip(plots.keys(), range(rows), range(cols)):
            print(plot, row+1, col+1)
            self.ci.addItem(plot, row+1, col+1)

    def get_plots(self):
        return self.ci.items
