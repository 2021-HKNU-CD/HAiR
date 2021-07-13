import time

import cv2
import qimage2ndarray

import sys

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UI파일 연결
# 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
from src.util.capture import Capture

form_class = uic.loadUiType("hair_interface.ui")[0]


class DisplayWorker(QThread):
    finished = pyqtSignal(QPixmap)

    def __init__(self):
        super(DisplayWorker, self).__init__()
        self.setTerminationEnabled(True)
        self.start()
        print("display worker is on")

    def run(self):
        while True:
            image = capture.get()
            image = qimage2ndarray.array2qimage(image)
            image = QPixmap.fromImage(image)
            self.finished.emit(image)
            time.sleep(0.2)


# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # displaying 용
        self.displayWorker = DisplayWorker()
        self.displayWorker.finished.connect(self.take_a_shot)

        # 닫기 버튼
        self.closeBtn.clicked.connect(self.close)

    @pyqtSlot(QPixmap)
    def take_a_shot(self, image):
        print('this is a take a shot')
        self.cameraInput.setPixmap(image.scaledToWidth(1280))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    capture = Capture(0)
    myWindow = WindowClass()
    myWindow.showFullScreen()

    images = next(os.walk(BASE_DIR + '/../../ref_images'), (None, None, []))[2]
    print(f"ref_images contains total of {len(images)}")

    try:
        app.exec_()
    finally:
        capture.destroy()
