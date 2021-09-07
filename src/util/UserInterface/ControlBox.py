from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel

RIGHT_BOX_WIDTH = 400
RIGHT_BOX_HEIGHT = int(RIGHT_BOX_WIDTH * 0.5625)


class ControlBox(QVBoxLayout):
    result = pyqtSignal()
    transform = pyqtSignal()
    close = pyqtSignal()

    def __init__(self):
        super(ControlBox, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.progress_label = QLabel()
        self.progress_label.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QFont()
        font.setWeight(64)
        font.setPointSize(18)
        self.progress_label.setFont(font)
        self.set_ready()

        self.transform_button = QPushButton("변환")
        self.transform_button.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.transform_button.clicked.connect(self.transform_signal)
        self.transform_button.setFont(font)

        self.result_button = QPushButton("결과보기")
        self.result_button.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.result_button.clicked.connect(self.result_signal)
        self.result_button.setFont(font)
        self.result_button.setDisabled(True)

        self.close_button = QPushButton("종료")
        self.close_button.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.close_button.clicked.connect(self.close_signal)
        self.close_button.setFont(font)

        self.addWidget(self.progress_label)
        self.addWidget(self.transform_button)
        self.addWidget(self.result_button)
        self.addWidget(self.close_button)

    def result_signal(self):
        self.result.emit()

    def transform_signal(self):
        self.transform.emit()

    def close_signal(self):
        self.close.emit()

    def set_ready(self):
        self.progress_label.setText("Ready")
        self.progress_label.setStyleSheet("background-color: green; color:white")

    def set_processing(self):
        self.progress_label.setText("PROCESSING")
        self.progress_label.setStyleSheet("background-color: blue; color:white")

    def set_error(self):
        self.progress_label.setText("ERROR")
        self.progress_label.setStyleSheet("background-color: red; color:white")

    def initialize(self):
        self.set_ready()
        self.result_button.setDisabled(True)
