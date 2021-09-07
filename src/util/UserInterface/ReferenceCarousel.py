from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QPushButton

REF_CARO_WIDTH = 350
REF_CARO_HEIGHT = int(REF_CARO_WIDTH * 0.5625)


class Reference(QLabel):
    selected_reference = pyqtSignal(int)

    def __init__(self, number: str, index: int, ref_images: list):
        super(Reference, self).__init__()
        self.ref_images = ref_images
        self.ref_index = None
        self.setText(number)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFixedWidth(REF_CARO_WIDTH)
        self.setFixedHeight(REF_CARO_HEIGHT)
        self.set_image(index)
        self.mousePressEvent = self.clicked

    def set_image(self, index: int) -> None:
        self.setPixmap(self.ref_images[index][1].scaledToHeight(REF_CARO_HEIGHT))
        self.ref_index = index

    def clicked(self, event):
        self.selected_reference.emit(self.ref_index)


class ReferenceCarousel(QHBoxLayout):
    def __init__(self, ref_images: list):
        super(ReferenceCarousel, self).__init__()

        self.ref_index = 0

        self.ref_images = ref_images
        self.REF_SIZE = len(ref_images)

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
            Reference("0", 0, self.ref_images),
            Reference("1", 1, self.ref_images),
            Reference("2", 2, self.ref_images),
            # Reference("3", 3, self.ref_images),
            # Reference("4", 4, self.ref_images)
        ]

        self.addWidget(self.left, 1)

        for R in self.carousel:
            self.addWidget(R, 2)

        self.addWidget(self.right, 1)

    def rotate_left(self):
        # 왼쪽으로 5개씩 회전
        self.right.setEnabled(True)
        if self.ref_index - 3 < 0:
            self.ref_index = 0
            self.left.setEnabled(False)
        else:
            self.ref_index -= 3
        [c.set_image(self.ref_index + i) for i, c in enumerate(self.carousel)]

    def rotate_right(self):
        # 오른쪽으로 5개씩 회전
        self.left.setEnabled(True)
        if self.ref_index + 2 + 3 >= self.REF_SIZE - 1:
            self.ref_index = self.REF_SIZE - 4
            self.right.setEnabled(False)
        else:
            self.ref_index += 3
        [c.set_image(self.ref_index + i) for i, c in enumerate(self.carousel)]
