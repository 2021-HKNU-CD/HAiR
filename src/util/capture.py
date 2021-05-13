import cv2
import numpy as np


class Capture:
    def __init__(self, device_num: int):
        self.capture_dev = cv2.VideoCapture(device_num)

    def get(self) -> np.ndarray:
        ret, image = self.capture_dev.read()
        return image


capture = Capture(0)
cv2.imwrite('test.jpg', capture.get())
