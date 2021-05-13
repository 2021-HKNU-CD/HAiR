from collections import OrderedDict

import cv2
import dlib
import numpy as np

FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])


class FaceFeature:
    def __init__(self):
        self.facial_features_cordinates = {}
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('checkpoints/shape_predictor_68_face_landmarks.dat')

    def get(self, image: np.ndarray):
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_scale, 1)
        rect = rects[0]
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray_scale, rect)
        shape = self.shape_to_numpy_array(shape)

        left, right, mouth = self.get_feature(shape)
        # print(left)
        # print(right)
        # print(mouth)
        # output = image
        # output = cv2.line(output, left, left, (0, 0, 255), 5)
        # output = cv2.line(output, right, right, (0, 0, 255), 5)
        # output = cv2.line(output, mouth, mouth, (0, 0, 255), 5)
        # cv2.imwrite(f'test{i}.jpg', output)
        return left, right, mouth

    def shape_to_numpy_array(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coordinates = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coordinates[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coordinates

    def get_feature(self, shape):
        left_start, left_end = FACIAL_LANDMARKS_INDEXES["Left_Eye"]
        right_start, right_end = FACIAL_LANDMARKS_INDEXES["Right_Eye"]
        mouth_start, mouth_end = FACIAL_LANDMARKS_INDEXES["Mouth"]
        left_pts = shape[left_start: left_end]
        right_pts = shape[right_start:right_end]
        mouth_pts = shape[mouth_start:mouth_end]
        center_left_eye = (sum([x for x, _ in left_pts]) // len(left_pts),
                           sum([y for _, y in left_pts]) // len(left_pts))
        center_right_eye = (sum([x for x, _ in right_pts]) // len(right_pts),
                            sum([y for _, y in right_pts]) // len(right_pts))
        center_mouth = (sum([x for x, _ in mouth_pts]) // len(mouth_pts),
                        sum([y for _, y in mouth_pts]) // len(mouth_pts))

        return center_left_eye, center_right_eye, center_mouth
        # print(FACIAL_LANDMARKS_INDEXES["Right_Eye"])
        # print(FACIAL_LANDMARKS_INDEXES["Mouth"])

        # return the output image
        # print(self.facial_features_cordinates)
        # return output


faceFeature = FaceFeature()
faceFeature.get(cv2.imread('../data/test/unnamed.jpg'))
