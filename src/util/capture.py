import cv2
import numpy as np


class Capture:
    def __init__(self, device_num: int):
        self.capture_dev = cv2.VideoCapture()
        self.capture_dev.open(device_num + cv2.CAP_DSHOW)

    def get(self) -> np.ndarray:
        ret, image = self.capture_dev.read()
        return image

    def destroy(self) -> None:
        self.capture_dev.release()
