import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

from src.util.UserInterface.ndarrayToQpixmap import ndarray_to_qpixmap

QR_RESULT_WIDTH = 1280
QR_RESULT_HEIGHT = int(QR_RESULT_WIDTH * 0.5625)


class Result(QWidget):
    qr_done = pyqtSignal()

    def __init__(self):
        super(Result, self).__init__()
        vertical = QVBoxLayout()
        vertical.setAlignment(QtCore.Qt.AlignCenter)

        self.result_image = QLabel("result_image")
        self.result_image.setAlignment(QtCore.Qt.AlignCenter)
        self.result_image.setFixedSize(QR_RESULT_WIDTH, QR_RESULT_HEIGHT)

        complete = QPushButton("Done")
        complete.clicked.connect(self.clicked)

        vertical.addWidget(self.result_image)
        vertical.addWidget(complete)

        self.setLayout(vertical)

    def clicked(self):
        self.qr_done.emit()

    def set(self, image: np.ndarray):
        self.result_image.setPixmap(
            ndarray_to_qpixmap(
                image
            ).scaledToWidth(QR_RESULT_WIDTH)
        )
