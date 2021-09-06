import numpy as np
import qimage2ndarray
from PyQt5.QtGui import QPixmap


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(
        qimage2ndarray
            .array2qimage(image)
            .rgbSwapped()
    )
