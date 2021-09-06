from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QLabel

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * 0.5625)


class Display(QVBoxLayout):
    def __init__(self):
        super(Display, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.image = QLabel("Display")
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setStyleSheet("background-color: black; color:white")
        self.image.setFixedWidth(DISPLAY_WIDTH)
        self.image.setFixedHeight(DISPLAY_HEIGHT)

        self.addWidget(self.image)

    def set_image(self, image: QPixmap) -> None:
        self.image.setPixmap(image.scaledToHeight(DISPLAY_HEIGHT))
