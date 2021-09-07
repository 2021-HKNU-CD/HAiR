from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QRadioButton


class RadioBox(QGroupBox):
    def __init__(self):
        super(RadioBox, self).__init__()
        self.type = "머리 색상"
        self.setAlignment(QtCore.Qt.AlignCenter)
        layout = QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)

        font = QFont()
        font.setPointSize(32)
        self.appearance = QRadioButton("머리 색상")
        self.appearance.setFixedHeight(60)
        self.appearance.setFont(font)
        self.appearance.setChecked(True)
        self.shapeStructure = QRadioButton("머리 모양")
        self.shapeStructure.setFixedHeight(60)
        self.shapeStructure.setFont(font)

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
