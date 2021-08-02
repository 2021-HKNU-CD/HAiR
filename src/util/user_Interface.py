import time

import cv2
import numpy as np
import qimage2ndarray

import sys

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
import os

from src.transformers.AppearanceTransformer import AppearanceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.util.capture import Capture

# UI파일 연결
# 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("hair_interface.ui")[0]

ref_images = next(os.walk(BASE_DIR + '/../../ref_images'), (None, None, []))[2]
print(f"ref_images contains total of {len(ref_images)}")

np_ref_images = []
for ref in ref_images:
    np_ref_images.append(cv2.imread(BASE_DIR + '/../../ref_images/' + ref))

Transformer = {
    'Appearance': AppearanceTransformer(),
    'ShapeStructure': ShapeStructureTransformer()
}


class DisplayWorker(QThread):
    finished = pyqtSignal(QPixmap)

    def __init__(self):
        super(DisplayWorker, self).__init__()
        self.setTerminationEnabled(True)
        self.start()
        print("display worker is on")

    def run(self):
        time.sleep(1)
        while True:
            image: np.ndarray = capture.get()
            t_image = Transformer['Appearance'].transform(image)
            qimage = qimage2ndarray.array2qimage(t_image)
            qimage = qimage.rgbSwapped()
            qimage = QPixmap.fromImage(qimage)
            self.finished.emit(qimage)
            time.sleep(0.1)


# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class):
    # reference carousel index
    ref_index = 0
    ref_mode = 'Appearance'
    selected_images = {
        'Appearance': -1,
        'Shape': -1,
        'Structure': -1
    }

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # reference_image를 QPixmap으로 변환
        self.qpixmap_ref_images = []
        for image_path in ref_images:
            qpixmap_var = QPixmap()
            qpixmap_var.load(BASE_DIR + '/../../ref_images/' + image_path)
            self.qpixmap_ref_images.append(qpixmap_var)

        # not selected image
        qp = QPixmap()
        qp.load(BASE_DIR + '/image_not_selected.png')
        self.qpixmap_ref_images.append(qp)

        # refresh selected reference
        self.refresh_window()

        # refresh carousel
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
        self.displayWorker.finished.connect(self.display)

        # selected reference
        self.windowAppearance.mousePressEvent = self.clicked_appearance
        self.windowShape.mousePressEvent = self.clicked_shape
        self.windowStructure.mousePressEvent = self.clicked_structure

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
        if self.ref_index != len(self.qpixmap_ref_images) - 6:
            self.ref_index += 1
        if self.ref_index < len(self.qpixmap_ref_images) - 6:
            self.reference_left.setEnabled(True)
        else:
            self.reference_right.setEnabled(False)
        self.show_ref(self.qpixmap_ref_images[self.ref_index: self.ref_index + 5])

    def clicked_reference1(self, event):
        self.selected_images[self.ref_mode] = self.ref_index + 0
        self.refresh_window()
        pass

    def clicked_reference2(self, event):
        self.selected_images[self.ref_mode] = self.ref_index + 1
        self.refresh_window()
        pass

    def clicked_reference3(self, event):
        self.selected_images[self.ref_mode] = self.ref_index + 2
        self.refresh_window()
        pass

    def clicked_reference4(self, event):
        self.selected_images[self.ref_mode] = self.ref_index + 3
        self.refresh_window()
        pass

    def clicked_reference5(self, event):
        self.selected_images[self.ref_mode] = self.ref_index + 4
        self.refresh_window()
        pass

    # transform type
    def transformTypeCheck(self):
        if self.radio_app.isChecked():
            self.ref_mode = "Appearance"

        elif self.radio_shape.isChecked():
            self.ref_mode = "Shape"

        elif self.radio_struct.isChecked():
            self.ref_mode = "Structure"

    # reference windows
    def refresh_window(self):
        self.windowAppearance.setPixmap(self.qpixmap_ref_images[self.selected_images["Appearance"]].scaledToWidth(370))
        if self.selected_images["Appearance"] != -1:
            Transformer["Appearance"].set_reference(np_ref_images[self.selected_images["Appearance"]])
        self.windowStructure.setPixmap(self.qpixmap_ref_images[self.selected_images["Structure"]].scaledToWidth(370))
        self.windowShape.setPixmap(self.qpixmap_ref_images[self.selected_images["Shape"]].scaledToWidth(370))

    def clicked_appearance(self, event):
        self.selected_images["Appearance"] = -1
        self.refresh_window()
        Transformer['Appearance'].set_reference(None)

    def clicked_shape(self, event):
        self.selected_images["Shape"] = -1
        self.refresh_window()
        Transformer['ShapeStructure'].set_reference(None)

    def clicked_structure(self, event):
        self.selected_images["Structure"] = -1
        self.refresh_window()
        Transformer['ShapeStructure'].set_reference(None)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    capture = Capture(0)
    myWindow = WindowClass()
    myWindow.showFullScreen()

    try:
        app.exec_()
    finally:
        capture.destroy()
