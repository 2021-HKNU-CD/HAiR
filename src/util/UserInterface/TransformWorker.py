import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable

from src.transformers.Transformer import Transformer
from src.util.capture import Capture


class TransformerSignal(QObject):
    transformed = pyqtSignal(np.ndarray)


class TransformWorker(QRunnable):
    def __init__(self, capture: Capture, transformer: Transformer):
        super(TransformWorker, self).__init__()
        self.signal = TransformerSignal()
        self.capture = capture
        self.T = transformer

    def run(self):
        image = self.capture.get()
        transformed_image = self.T.transform(image)
        if (transformed_image == image).all():
            self.signal.transformed.emit(np.ndarray([0]))

        else:
            self.signal.transformed.emit(transformed_image)
