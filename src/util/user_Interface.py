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

ref_images = next(os.walk(BASE_DIR + '/../../ref_images'), (None, None, []))[2]
print(f"ref_images contains total of {len(ref_images)}")


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
            image = image.rgbSwapped()
            image = QPixmap.fromImage(image)
            self.finished.emit(image)
            time.sleep(0.1)


# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class):
    # reference carousel index
    ref_index = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # reference_image를 QPixmap으로 변환
        self.qpixmap_ref_images = []
        for image_path in ref_images:
            qpixmap_var = QPixmap()
            qpixmap_var.load(BASE_DIR + '/../../ref_images/' + image_path)
            self.qpixmap_ref_images.append(qpixmap_var)

        self.show_ref(self.qpixmap_ref_images[:5])

        # reference carousel control
        self.reference_left.setEnabled(False)
        self.reference_left.clicked.connect(self.ref_rotate_left)
        self.reference_right.clicked.connect(self.ref_rotate_right)

        self.reference1.mousePressEvent = self.clicked_reference1
        self.reference2.mousePressEvent = self.clicked_reference2
        self.reference3.mousePressEvent = self.clicked_reference3
        self.reference4.mousePressEvent = self.clicked_reference4
        self.reference5.mousePressEvent = self.clicked_reference5

        # transform image selection
        self.radio_app.clicked.connect(self.transformTypeCheck)
        self.radio_shape.clicked.connect(self.transformTypeCheck)
        self.radio_struct.clicked.connect(self.transformTypeCheck)

        # displaying 용
        self.displayWorker = DisplayWorker()
        self.displayWorker.finished.connect(self.take_a_shot)

        # 닫기 버튼
        self.closeBtn.clicked.connect(self.close)

    @pyqtSlot(QPixmap)
    def display(self, image: QPixmap):
        self.cameraInput.setPixmap(image.scaledToWidth(1280))

    # reference carousel
    def show_ref(self, images):
        self.reference1.setPixmap(images[0].scaledToHeight(171))
        self.reference2.setPixmap(images[1].scaledToHeight(171))
        self.reference3.setPixmap(images[2].scaledToHeight(171))
        self.reference4.setPixmap(images[3].scaledToHeight(171))
        self.reference5.setPixmap(images[4].scaledToHeight(171))

    def ref_rotate_left(self):
        if self.ref_index != 0:
            self.ref_index -= 1
        if self.ref_index > 0:
            self.reference_right.setEnabled(True)
        else:
            self.reference_left.setEnabled(False)
        self.show_ref(self.qpixmap_ref_images[self.ref_index: self.ref_index + 5])

    def ref_rotate_right(self):
        if self.ref_index != len(self.qpixmap_ref_images) - 5:
            self.ref_index += 1
        if self.ref_index < len(self.qpixmap_ref_images) - 5:
            self.reference_left.setEnabled(True)
        else:
            self.reference_right.setEnabled(False)
        self.show_ref(self.qpixmap_ref_images[self.ref_index: self.ref_index + 5])

    def clicked_reference1(self, event):
        pass

    def clicked_reference2(self, event):
        pass

    def clicked_reference3(self, event):
        pass

    def clicked_reference4(self, event):
        pass

    def clicked_reference5(self, event):
        pass

    # transform type
    def transformTypeCheck(self):
        if self.radio_app.isChecked():
            print('apppearance')
        elif self.radio_shape.isChecked():
            print("shape")
        elif self.radio_struct.isChecked():
            print("structure")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    capture = Capture(0)
    myWindow = WindowClass()
    myWindow.showFullScreen()

    try:
        app.exec_()
    finally:
        capture.destroy()
