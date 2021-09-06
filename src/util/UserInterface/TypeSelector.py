import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget


def set_align_center(x: QWidget) -> QWidget:
    x.setAlignment(QtCore.Qt.AlignCenter)
    return x


RIGHT_BOX_WIDTH = 400
RIGHT_BOX_HEIGHT = int(RIGHT_BOX_WIDTH * 0.5625)


class ReferenceViewer(QVBoxLayout):
    unselect = pyqtSignal(str)

    def __init__(self, label: str, description: str, ref_images: list):
        super(ReferenceViewer, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.ref_images = ref_images

        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setWeight(16)

        description_font = QFont()
        description_font.setPointSize(20)
        description_font.setWeight(12)

        self.label = set_align_center(QLabel(label))
        self.label.setFont(title_font)
        self.description = set_align_center(QLabel(description))
        self.description.setFont(description_font)

        self.image = QLabel(label)
        self.image.setFixedWidth(RIGHT_BOX_WIDTH)
        self.image.setFixedHeight(RIGHT_BOX_HEIGHT)
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.mousePressEvent = self.clicked

        self.set_image(-1)
        self.ref_index = None

        self.addWidget(self.label, 1)
        self.addWidget(self.description, 1)
        self.addWidget(self.image, 4)

    def set_image(self, index: int):
        """
        뷰어에 이미지를 설정하고 그에 따른 인덱스도 설정
        set a reference image to the viewer and set a index
        @param index: index of reference image
        """
        self.image.setPixmap(self.ref_images[index][1].scaledToHeight(RIGHT_BOX_HEIGHT))
        self.ref_index = index

    def clicked(self, event):
        self.set_image(-1)
        self.unselect.emit(self.image.text())


class TypeSelector(QVBoxLayout):
    def __init__(self, ref_images: list):
        super(TypeSelector, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.selectors = {
            "머리 색상": ReferenceViewer('머리 색상', "머리의 색상을 변화시킵니다.\n(검은색 -> 노란색)\n클릭시 해제", ref_images),
            "머리 모양": ReferenceViewer("머리 모양", "머리의 형태를 변화시킵니다.\n(짧은 머리 -> 긴머리)\n클릭시 해제", ref_images),
        }
        [self.addLayout(value, 1) for value in self.selectors.values()]

    def set_reference(self, ref_type: str, index: int) -> None:
        """
        선택자들에게 레퍼런스 이미지 전달
        set a reference image to the selectors
        @param ref_type: type of reference ('Appearance' | "ShapeStructure")
        @param index: index of reference (-1) means not selected
        """
        self.selectors[ref_type].set_image(index)

    def get_ref_index(self, ref_type: str) -> int:
        """
        레퍼런스 타입에 맞는 이미지의 인덱스를 반환
        returns index of selected reference type
        @param ref_type: type of reference ('Appearance' | "ShapeStructure")
        @return: index of selected reference type. if not selected (-1)
        """
        return self.selectors[ref_type].ref_index

    def initialize(self):
        for referenceViewer in self.selectors.values():
            referenceViewer.unselect.emit(referenceViewer.image.text())
            referenceViewer.set_image(-1)
