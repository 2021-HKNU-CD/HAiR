import cv2
import numpy as np

capture_dev = cv2.VideoCapture(0)


class Capture:
    def __init__(self, device_num: int):
        self.capture_dev = cv2.VideoCapture(device_num)

    def get_image(self) -> np.ndarray:
        return self.capture_dev.read()
