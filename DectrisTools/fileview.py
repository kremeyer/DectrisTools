def run():
    import sys
    import pyqtgraph as pg
    from .ui.fileview import FileViewUi
    from PyQt5 import QtWidgets

    pg.setConfigOption("background", "k")
    pg.setConfigOption("foreground", "w")

    app = QtWidgets.QApplication(sys.argv)
    ui = FileViewUi()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
