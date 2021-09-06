from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton


class StartScreen(QVBoxLayout):
    start = pyqtSignal()
    close = pyqtSignal()

    def __init__(self):
        super(StartScreen, self).__init__()
        self.welcome = QLabel("안녕하세요. HAiR 입니다.")

        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start_clicked)
        self.close_button = QPushButton("종료")
        self.close_button.clicked.connect(self.close_clicked)

        self.setAlignment(QtCore.Qt.AlignCenter)

        self.addWidget(self.welcome)
        self.addWidget(self.start_button)
        self.addWidget(self.close_button)

    def start_clicked(self):
        self.start.emit()

    def close_clicked(self):
        self.close.emit()
