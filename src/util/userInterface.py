import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSlot, QThreadPool, QTimer
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *

from src.components.BoundingBox.BoundingBox import BoundingBox
from src.transformers.Transformer import Transformer, getTransformer
from src.util.UserInterface.ControlBox import ControlBox
from src.util.UserInterface.Display import Display
from src.util.UserInterface.DisplayWorker import DisplayWorker
from src.util.UserInterface.RadioBox import RadioBox
from src.util.UserInterface.ReferenceCarousel import ReferenceCarousel
from src.util.UserInterface.Result import Result
from src.util.UserInterface.StartScreen import StartScreen
from src.util.UserInterface.TransformWorker import TransformWorker
from src.util.UserInterface.TypeSelector import TypeSelector
from src.util.UserInterface.ndarrayToQpixmap import ndarray_to_qpixmap
from src.util.capture import Capture

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ref_images = next(os.walk(BASE_DIR + '/../../ref_images'), (None, None, []))[2]
NOT_FOUND: QPixmap
T: Transformer


def set_align_center(x: QWidget) -> QWidget:
    x.setAlignment(QtCore.Qt.AlignCenter)
    return x


def get_qimage(path: str) -> QPixmap:
    from src.util.UserInterface.ndarrayToQpixmap import ndarray_to_qpixmap
    np_image = cv2.imread(path)
    bounding_box = BoundingBox(np_image)
    try:
        bounding_box.get_bounding_box()
    except ValueError as V:
        print(f"when processing {path} {V}")
    except TypeError as T:
        print(f"when processing {path} {T}")
    return ndarray_to_qpixmap(bounding_box.get_origin_patch())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.window_stack = QStackedWidget(self)
        self.start_screen = StartScreen()
        self.display_worker = DisplayWorker(capture)
        self.display = Display()
        self.radio_box = RadioBox()
        self.reference_carousel = ReferenceCarousel(ref_images)
        self.control_box = ControlBox()
        self.type_selector = TypeSelector(ref_images)
        self.result = Result()
        self.transform_worker = TransformWorker(capture, T)

        self.setWindowTitle("HAiR")
        self.setGeometry(0, 0, 1920, 1080)
        self.setup()

    @pyqtSlot()
    def start_signal(self):
        self.window_stack.setCurrentIndex(1)
        self.type_selector.initialize()
        self.control_box.initialize()
        self.display_worker.go = True
        self.display_worker.start()

    @pyqtSlot()
    def close_signal(self):
        self.close()

    @pyqtSlot()
    def result_signal(self):
        # deprecated dont use
        self.window_stack.setCurrentIndex(2)
        self.display_worker.go = False

    @pyqtSlot(int)
    def ref_select(self, index: int):
        self.type_selector.set_reference(self.radio_box.type, index)
        if self.radio_box.type == "머리 색상":
            T.set_appearance_ref(ref_images[index][0])
        else:
            T.set_shape_ref(ref_images[index][0])
            T.set_structure_ref(ref_images[index][0])

    @pyqtSlot(str)
    def ref_unselect(self, ref_type: str) -> None:
        if ref_type == "머리 색상":
            T.set_appearance_ref(None)
        else:
            T.set_shape_ref(None)
            T.set_structure_ref(None)

    @pyqtSlot(QPixmap)
    def get_image(self, image: QPixmap):
        self.display.set_image(image)

    @pyqtSlot()
    def back_to_start_signal(self):
        self.window_stack.setCurrentIndex(0)

    @pyqtSlot()
    def qr_done_signal(self):
        self.window_stack.setCurrentIndex(0)

    @pyqtSlot(int)
    def result_clicked_signal(self, timestamp: int):
        self.qr_result.set(timestamp)
        self.window_stack.setCurrentIndex(3)

    @pyqtSlot()
    def transform_signal(self):
        self.control_box.transform_button.setDisabled(True)
        self.control_box.set_processing()

        pool = QThreadPool.globalInstance()
        pool.start(self.transform_worker)
        self.transform_worker = TransformWorker(capture, transformer=T)
        self.transform_worker.signal.transformed.connect(self.transformed_signal)

    @pyqtSlot(np.ndarray)
    def transformed_signal(self, image: np.ndarray):
        if image.ndim == 1:
            # when failed
            self.control_box.set_error()
            QTimer().singleShot(2000, self.control_box.set_ready)
        else:
            self.control_box.set_ready()
            self.control_box.result_button.setDisabled(False)
            self.result.set(image)

        self.control_box.transform_button.setDisabled(False)

    def setup(self):
        # Start Screen
        self.start_screen.start.connect(self.start_signal)
        self.start_screen.close.connect(self.close_signal)

        # DISPLAY
        self.display_worker.finished.connect(self.get_image)

        # REF CAROUSEL
        [i.selected_reference.connect(self.ref_select) for i in self.reference_carousel.carousel]

        # TYPE SELECTOR
        [i.unselect.connect(self.ref_unselect) for i in self.type_selector.selectors.values()]

        # CONTROL BOX
        self.control_box.result.connect(self.result_signal)
        self.control_box.transform.connect(self.transform_signal)
        self.control_box.close.connect(self.close_signal)

        # QR result
        self.result.qr_done.connect(self.qr_done_signal)

        # Transform thread
        self.transform_worker.signal.transformed.connect(self.transformed_signal)

        # setup UI
        start = QWidget(self)
        start.setLayout(self.start_screen)

        self.setCentralWidget(self.window_stack)

        transform = QWidget(self)
        transform_window = set_align_center(QHBoxLayout())
        left_box = set_align_center(QVBoxLayout())
        right_box = set_align_center(QVBoxLayout())

        left_box.addLayout(self.display, 1)
        left_box.addWidget(self.radio_box)
        left_box.addLayout(self.reference_carousel, 1)

        right_box.addLayout(self.type_selector, 3)
        right_box.addLayout(self.control_box, 1)

        transform_window.addStretch(1)
        transform_window.addLayout(left_box, 8)
        transform_window.addLayout(right_box, 4)
        transform.setLayout(transform_window)

        self.window_stack.addWidget(start)  # 0
        self.window_stack.addWidget(transform)  # 1
        self.window_stack.addWidget(self.result)  # 2


if __name__ == "__main__":
    T = getTransformer()
    capture = Capture(0)
    app = QApplication(sys.argv)

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
            ndarray_to_qpixmap(cv2.imread(BASE_DIR + '/image_not_selected.png'))
        ]
    )

    mainWindow = MainWindow()
    mainWindow.showFullScreen()
    ret = app.exec_()
    sys.exit(ret)
