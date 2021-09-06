import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from src.components.Aligner.AlignerWing import AlignerWing
from src.util.UserInterface.ndarrayToQpixmap import ndarray_to_qpixmap


class DisplayWorker(QThread):
    finished = pyqtSignal(QPixmap)

    def __init__(self, capture):
        super(DisplayWorker, self).__init__()
        self.setTerminationEnabled(True)
        self.go = True
        self.capture = capture

    def run(self) -> None:
        while self.go:
            image: np.ndarray = self.capture.get()

            for x, y in CELEB_REF:
                image = cv2.line(image, (OFFSET_X + x, OFFSET_Y + y), (OFFSET_X + x, OFFSET_Y + y), (255, 0, 0), 5)
            # t_image = T.transform(image)

            self.finished.emit(ndarray_to_qpixmap(image))


CELEB_REF = list(map(lambda y: (int(y[0]), int(y[1])), AlignerWing.FaceAligner.CELEB_REF))
OFFSET_Y = 52
OFFSET_X = 384
