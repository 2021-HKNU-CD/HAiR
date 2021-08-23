import os
import sys

import cv2
import numpy as np
import qimage2ndarray
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *

from src.transformers.Transformer import Transformer, getTransformer
from src.util.capture import Capture

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ref_images = next(os.walk(BASE_DIR + '/../../ref_images'), (None, None, []))[2]
REF_SIZE = len(ref_images)
NOT_FOUND: QPixmap
T: Transformer = None

SIZE_POLICY = QSizePolicy.Ignored

RIGHT_BOX_WIDTH = 400
RIGHT_BOX_HEIGHT = int(RIGHT_BOX_WIDTH * 0.5625)  # 16:9 ratio

DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * 0.5625)

REF_CARO_WIDTH = 250
REF_CARO_HEIGHT = int(REF_CARO_WIDTH * 0.5625)


def set_centered_black_white(x: QWidget) -> QWidget:
    x.setStyleSheet("background-color: black; color:white")
    x.setAlignment(QtCore.Qt.AlignCenter)
    return x


def set_align_center(x: QWidget) -> QWidget:
    x.setAlignment(QtCore.Qt.AlignCenter)
    return x


def get_qimage(path: str) -> QPixmap:
    qimage = QPixmap()
    qimage.load(path, flags=QtCore.Qt.AutoColor)
    return qimage


class DisplayWorker(QThread):
    finished = pyqtSignal(QPixmap)

    def __init__(self):
        super(DisplayWorker, self).__init__()
        self.setTerminationEnabled(True)
        self.start()

    def run(self) -> None:
        while T:
            image: np.ndarray = capture.get()
            t_image = T.transform(image)
            self.finished.emit(
                QPixmap.fromImage(
                    qimage2ndarray
                        .array2qimage(t_image)
                        .rgbSwapped()
                )
            )


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


class RadioBox(QGroupBox):
    def __init__(self):
        super(RadioBox, self).__init__()
        self.type = "Appearance"
        self.setAlignment(QtCore.Qt.AlignCenter)
        layout = QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.appearance = QRadioButton("Appearance")
        self.appearance.setChecked(True)
        self.shape = QRadioButton("Shape")
        self.structure = QRadioButton("Structure")

        layout.addWidget(self.appearance)
        layout.addWidget(self.shape)
        layout.addWidget(self.structure)

        [i.clicked.connect(self.clicked) for i in [self.appearance, self.shape, self.structure]]

        self.setLayout(layout)
        self.setTitle('타입을 고르세요!')

    def clicked(self):
        self.type = "".join(
            map(lambda x: x.text(),
                filter(lambda x: x.isChecked(),
                       [self.appearance, self.shape, self.structure])))


class Reference(QLabel):
    selected_reference = pyqtSignal(int)

    def __init__(self, number: str, index: int):
        super(Reference, self).__init__()
        self.ref_index = None
        self.setText(number)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFixedWidth(REF_CARO_WIDTH)
        self.setFixedHeight(REF_CARO_HEIGHT)
        self.set_image(index)
        self.mousePressEvent = self.clicked

    def set_image(self, index: int) -> None:
        self.setPixmap(ref_images[index][1].scaledToHeight(REF_CARO_HEIGHT))
        self.ref_index = index

    def clicked(self, event):
        self.selected_reference.emit(self.ref_index)


class ReferenceCarousel(QHBoxLayout):
    def __init__(self):
        super(ReferenceCarousel, self).__init__()

        self.ref_index = 0

        self.setAlignment(QtCore.Qt.AlignCenter)

        self.left = QPushButton("LEFT")
        self.right = QPushButton("RIGHT")

        self.left.setFixedWidth(75)
        self.left.setFixedHeight(REF_CARO_HEIGHT)
        self.right.setFixedWidth(75)
        self.right.setFixedHeight(REF_CARO_HEIGHT)
        self.left.clicked.connect(self.rotate_left)
        self.right.clicked.connect(self.rotate_right)
        self.left.setEnabled(False)

        self.carousel = [
            Reference("0", 0),
            Reference("1", 1),
            Reference("2", 2),
            Reference("3", 3),
            Reference("4", 4)
        ]

        self.addWidget(self.left, 1)

        for R in self.carousel:
            self.addWidget(R, 2)

        self.addWidget(self.right, 1)

    def rotate_left(self):
        # 왼쪽으로 5개씩 회전
        self.right.setEnabled(True)
        if self.ref_index - 5 < 0:
            self.ref_index = 0
            self.left.setEnabled(False)
        else:
            self.ref_index -= 5
        [c.set_image(self.ref_index + i) for i, c in enumerate(self.carousel)]

    def rotate_right(self):
        # 오른쪽으로 5개씩 회전
        self.left.setEnabled(True)
        if self.ref_index + 4 + 5 >= REF_SIZE:
            self.ref_index = REF_SIZE - 5
            self.right.setEnabled(False)
        else:
            self.ref_index += 5
        [c.set_image(self.ref_index + i) for i, c in enumerate(self.carousel)]


class ReferenceViewer(QVBoxLayout):
    def __init__(self, label: str):
        super(ReferenceViewer, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.label = set_align_center(QLabel(label))

        self.image = QLabel(label)
        self.image.setFixedWidth(RIGHT_BOX_WIDTH)
        self.image.setFixedHeight(RIGHT_BOX_HEIGHT)
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.mousePressEvent = self.clicked

        self.set_image(-1)
        self.ref_index = None

        self.addWidget(self.label)
        self.addWidget(self.image, 2)

    def set_image(self, index: int):
        """
        뷰어에 이미지를 설정하고 그에 따른 인덱스도 설정
        set a reference image to the viewer and set a index
        @param index: index of reference image
        """
        self.image.setPixmap(ref_images[index][1].scaledToHeight(RIGHT_BOX_HEIGHT))
        self.ref_index = index

    def clicked(self, event):
        self.set_image(-1)
        T.set_shape_ref(self.label.text())
        if self.image.text() == "Appearance":
            T.set_appearance_ref(None)
        elif self.image.text() == "Shape":
            T.set_shape_ref(None)
        else:
            T.set_structure_ref(None)


class TypeSelector(QVBoxLayout):
    def __init__(self):
        super(TypeSelector, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.selectors = {
            "Appearance": ReferenceViewer('Appearance'),
            "Shape": ReferenceViewer("Shape"),
            "Structure": ReferenceViewer("Structure")
        }
        [self.addLayout(value, 1) for value in self.selectors.values()]

    def set_reference(self, ref_type: str, index: int) -> None:
        """
        선택자들에게 레퍼런스 이미지 전달
        set a reference image to the selectors
        @param ref_type: type of reference ('Appearance' | "Shape" | "Structure")
        @param index: index of reference (-1) means not selected
        """
        self.selectors[ref_type].set_image(index)

    def get_ref_index(self, ref_type: str) -> int:
        """
        레퍼런스 타입에 맞는 이미지의 인덱스를 반환
        returns index of selected reference type
        @param ref_type: type of reference ('Appearance' | "Shape" | "Structure")
        @return: index of selected reference type. if not selected (-1)
        """
        return self.selectors[ref_type].ref_index


class ControlBox(QVBoxLayout):
    close = pyqtSignal()

    def __init__(self):
        super(ControlBox, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.completed_button = QPushButton("complete")
        self.completed_button.setFixedSize(RIGHT_BOX_WIDTH, 40)

        self.close_button = QPushButton("close")
        self.close_button.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.close_button.clicked.connect(self.close)

        self.addWidget(self.completed_button)
        self.addWidget(self.close_button)

    def close_signal(self):
        self.close.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.display = Display()
        self.radio_box = RadioBox()
        self.reference_carousel = ReferenceCarousel()
        self.control_box = ControlBox()
        self.type_selector = TypeSelector()

        self.setWindowTitle("HAiR")
        self.setGeometry(0, 0, 1920, 1080)
        self.setup()

    @pyqtSlot()
    def close_signal(self):
        self.close()

    @pyqtSlot(int)
    def ref_select(self, index: int):
        self.type_selector.set_reference(self.radio_box.type, index)
        if self.radio_box.type == "Appearance":
            T.set_appearance_ref(ref_images[index][0])
        elif self.radio_box.type == "Shape":
            T.set_shape_ref(ref_images[index][0])
        else:
            T.set_structure_ref(ref_images[index][0])

    @pyqtSlot(QPixmap)
    def get_image(self, image: QPixmap):
        self.display.set_image(image)

    def setup(self):
        # DISPLAY
        # self.display.set_image(ref_images[2][1])
        self.displayWorker = DisplayWorker()
        self.displayWorker.finished.connect(self.get_image)

        # REF CAROUSEL
        [i.selected_reference.connect(self.ref_select) for i in self.reference_carousel.carousel]

        # CONTROL BOX
        self.control_box.close.connect(self.close_signal)

        # setup UI
        window = QWidget(self)
        self.setCentralWidget(window)

        horizontal_box = QHBoxLayout()
        left_box = QVBoxLayout()
        right_box = QVBoxLayout()

        left_box.addLayout(self.display, 1)
        left_box.addWidget(self.radio_box)
        left_box.addLayout(self.reference_carousel, 1)

        right_box.addLayout(self.type_selector)
        right_box.addLayout(self.control_box, 1)

        horizontal_box.addLayout(left_box, 3)
        horizontal_box.addLayout(right_box, 1)

        window.setLayout(horizontal_box)


if __name__ == "__main__":
    T = getTransformer()
    capture = Capture(1)
    app = QApplication(sys.argv)

    print(ref_images)

    ref_images = list(
        map(lambda x:
            [
                cv2.imread(BASE_DIR + '/../../ref_images/' + x),
                get_qimage(BASE_DIR + '/../../ref_images/' + x)
            ],
            ref_images)
    )

    ref_images.append(
        [
            cv2.imread(BASE_DIR + '/image_not_selected.png'),
            get_qimage(BASE_DIR + '/image_not_selected.png')
        ]
    )

    mainWindow = MainWindow()
    mainWindow.showFullScreen()
    ret = app.exec_()
    sys.exit(ret)
