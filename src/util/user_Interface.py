import os
import sys
import time

import cv2
import numpy as np
import qimage2ndarray
import qrcode
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject, QRunnable, QThreadPool
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
from pyngrok import ngrok, conf

from src.transformers.Transformer import Transformer, getTransformer
from src.util.capture import Capture

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ref_images = next(os.walk(BASE_DIR + '/../../ref_images'), (None, None, []))[2]
REF_SIZE = len(ref_images)
NOT_FOUND: QPixmap
T: Transformer = None

# NGROK
# PUBLIC_URL = ""
#
#
# def log_event_callback(log):
#     print(str(log))
#
#
# conf.get_default().log_event_callback = log_event_callback
# NGROK END

SIZE_POLICY = QSizePolicy.Ignored

RIGHT_BOX_WIDTH = 400
RIGHT_BOX_HEIGHT = int(RIGHT_BOX_WIDTH * 0.5625)  # 16:9 ratio

DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * 0.5625)

REF_CARO_WIDTH = 250
REF_CARO_HEIGHT = int(REF_CARO_WIDTH * 0.5625)

QR_WIDTH = 400
QR_HEIGHT = 400

QR_RESULT_WIDTH = 800
QR_RESULT_HEIGHT = int(QR_RESULT_WIDTH * 0.5625)


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


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(
        qimage2ndarray
            .array2qimage(image)
            .rgbSwapped()
    )


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


class DisplayWorker(QThread):
    finished = pyqtSignal(QPixmap)

    def __init__(self):
        super(DisplayWorker, self).__init__()
        self.setTerminationEnabled(True)
        self.go = True
        self.image = None

    def run(self) -> None:
        while self.go:
            self.image: np.ndarray = capture.get()
            # t_image = T.transform(image)

            self.finished.emit(ndarray_to_qpixmap(self.image))


class TransformerSignal(QObject):
    transformed = pyqtSignal(np.ndarray)


class TransformWorker(QRunnable):
    def __init__(self):
        super(TransformWorker, self).__init__()
        self.signal = TransformerSignal()

    def run(self):
        print('transform runnable')
        image = capture.get()
        transformed_image = T.transform(image)
        if (transformed_image == image).all():
            self.signal.transformed.emit(np.ndarray([0]))
        self.signal.transformed.emit(transformed_image)


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
        self.shapeStructure = QRadioButton("ShapeStructure")

        layout.addWidget(self.appearance)
        layout.addWidget(self.shapeStructure)

        [i.clicked.connect(self.clicked) for i in [self.appearance, self.shapeStructure]]

        self.setLayout(layout)
        self.setTitle('타입을 고르세요!')

    def clicked(self):
        self.type = "".join(
            map(lambda x: x.text(),
                filter(lambda x: x.isChecked(),
                       [self.appearance, self.shapeStructure])))


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
        else:
            T.set_shape_ref(None)
            T.set_structure_ref(None)


class TypeSelector(QVBoxLayout):
    def __init__(self):
        super(TypeSelector, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.selectors = {
            "Appearance": ReferenceViewer('Appearance'),
            "ShapeStructure": ReferenceViewer("ShapeStructure"),

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
    transform = pyqtSignal()

    def __init__(self):
        super(ControlBox, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.transform_button = QPushButton("transform")
        self.transform_button.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.transform_button.clicked.connect(self.transform)

        self.close_button = QPushButton("close")
        self.close_button.setFixedSize(RIGHT_BOX_WIDTH, 40)
        self.close_button.clicked.connect(self.close)

        self.addWidget(self.transform_button)
        self.addWidget(self.close_button)

    def close_signal(self):
        self.close.emit()

    def result_signal(self):
        self.transform.emit()


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


class Transforming(QWidget):
    def __init__(self):
        super(Transforming, self).__init__()
        vertical = QVBoxLayout()
        vertical.setAlignment(QtCore.Qt.AlignCenter)

        transforming = QLabel("TRANSFORMING")
        transforming.setAlignment(QtCore.Qt.AlignCenter)
        vertical.addWidget(transforming)
        self.setLayout(vertical)


class Retry(QWidget):
    def __init__(self):
        super(Retry, self).__init__()
        vertical = QVBoxLayout()
        vertical.setAlignment(QtCore.Qt.AlignCenter)

        transforming = QLabel("RETRY")
        transforming.setAlignment(QtCore.Qt.AlignCenter)
        vertical.addWidget(transforming)
        self.setLayout(vertical)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_stack = QStackedWidget(self)
        self.start_screen = StartScreen()
        self.display_worker = DisplayWorker()
        self.display = Display()
        self.radio_box = RadioBox()
        self.reference_carousel = ReferenceCarousel()
        self.control_box = ControlBox()
        self.type_selector = TypeSelector()
        self.result = Result()
        self.transforming = Transforming()
        self.transform_worker = TransformWorker()
        self.retry = Retry()

        self.setWindowTitle("HAiR")
        self.setGeometry(0, 0, 1920, 1080)
        self.setup()

    @pyqtSlot()
    def start_signal(self):
        self.window_stack.setCurrentIndex(1)
        self.display_worker.go = True
        self.display_worker.start()

    @pyqtSlot()
    def close_signal(self):
        self.close()

    @pyqtSlot(int)
    def ref_select(self, index: int):
        self.type_selector.set_reference(self.radio_box.type, index)
        if self.radio_box.type == "Appearance":
            T.set_appearance_ref(ref_images[index][0])
        else:
            T.set_shape_ref(ref_images[index][0])
            T.set_structure_ref(ref_images[index][0])

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
        self.window_stack.setCurrentIndex(3)
        self.display_worker.go = False

        pool = QThreadPool.globalInstance()
        pool.start(self.transform_worker)
        self.transform_worker = TransformWorker()
        self.transform_worker.signal.transformed.connect(self.transformed_signal)

    @pyqtSlot(np.ndarray)
    def transformed_signal(self, image: np.ndarray):

        try:
            self.result.set(image)
            self.window_stack.setCurrentIndex(2)
        except ValueError as v:
            print(v)
            self.window_stack.setCurrentIndex(4)
            time.sleep(2)
            self.window_stack.setCurrentIndex(1)
            self.display_worker.go = True
            self.display_worker.start()

        pass

    def setup(self):
        # Start Screen
        self.start_screen.start.connect(self.start_signal)
        self.start_screen.close.connect(self.close_signal)

        # DISPLAY
        self.display_worker.finished.connect(self.get_image)

        # REF CAROUSEL
        [i.selected_reference.connect(self.ref_select) for i in self.reference_carousel.carousel]

        # CONTROL BOX
        self.control_box.close.connect(self.close_signal)
        self.control_box.transform.connect(self.transform_signal)

        # QR result
        self.result.qr_done.connect(self.qr_done_signal)

        # Transform thread
        self.transform_worker.signal.transformed.connect(self.transformed_signal)

        # setup UI
        start = QWidget(self)
        start.setLayout(self.start_screen)

        self.setCentralWidget(self.window_stack)

        transform = QWidget(self)
        transform_window = QHBoxLayout()
        left_box = QVBoxLayout()
        right_box = QVBoxLayout()

        left_box.addLayout(self.display, 1)
        left_box.addWidget(self.radio_box)
        left_box.addLayout(self.reference_carousel, 1)

        right_box.addLayout(self.type_selector)
        right_box.addLayout(self.control_box, 1)

        transform_window.addLayout(left_box, 3)
        transform_window.addLayout(right_box, 1)
        transform.setLayout(transform_window)

        self.window_stack.addWidget(start)  # 0
        self.window_stack.addWidget(transform)  # 1
        self.window_stack.addWidget(self.result)  # 2
        self.window_stack.addWidget(self.transforming)  # 3
        self.window_stack.addWidget(self.retry)  # 4


if __name__ == "__main__":
    T = getTransformer()
    capture = Capture(0)
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

    # PUBLIC_URL = ngrok.connect("file://" + BASE_DIR + '/../../generated').public_url

    mainWindow = MainWindow()
    mainWindow.showFullScreen()
    ret = app.exec_()
    # ngrok.disconnect(PUBLIC_URL)
    T.clear()
    sys.exit(ret)
