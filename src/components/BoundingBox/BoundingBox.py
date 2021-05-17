import cv2
import numpy as np
from numpy.lib import math

from models.faceFeature import FaceFeature


class BoundingBox:
    """
    origin_image로 부터 눈, 코, 입 좌표를 추출하고
    얼굴 중심을 구한 뒤 회전각을 구하고 Margin을 구해서 origin_image에서 bounding box를 구합니다.
    """

    def __init__(self, original_image: np.ndarray, facefeat_model_path: str):
        '''
        param original_image : 1920 * 1080 크기의 카메라로 들어온 입력 이미지
        '''
        self.original_image = original_image
        self.margin = None
        self.theta = None
        self.face_center = None
        self.faceFeat = FaceFeature(model_path=facefeat_model_path)

    def get_bounding_box(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        1920 * 1080 원본 이미지에서 origin_patch 영역의 좌상단, 우상단, 좌하단, 우하단 영역을 반환합니다.
        """

        # BoundingBox = 얼굴 중심 ~ Margin*(sin(회전각) + cos(회전각))
        left_top, right_top, left_bottom, right_bottom = 0, 0, 0, 0

        width, height = len(self.original_image), len(self.original_image[0])

        # 왼쪽눈, 오른쪽눈, 입의 위치 파악
        left_eye, right_eye, mouth = self.faceFeat.get(self.original_image)

        # 얼굴의 중심
        self.face_center = (sum((left_eye[0], right_eye[0], mouth[0])) // 3,
                            sum((left_eye[1], right_eye[1], mouth[1])) // 3)

        middle_of_face = self.face_center

        # 두 눈을 이용해서 얼굴의 회전각파악
        x1 = left_eye[0]
        y1 = left_eye[1]
        x2 = right_eye[0]
        y2 = right_eye[1]
        m = (y1 - y2) / (x1 - x2)
        rotation = np.rad2deg(math.atan(m))
        self.theta = rotation

        # Margin
        distance_between_eyes = abs(y1 - y2) + abs(x1 - x2)
        self.margin = distance_between_eyes * 1.5

        # top 얼굴의 위
        top = (int(middle_of_face[0] - (self.margin * math.sin(-m))),
               int(middle_of_face[1] - (self.margin * math.cos(-m))))

        # bottom 얼굴의 아래
        bottom = (int(middle_of_face[0] + (self.margin * math.sin(-m))),
                  int(middle_of_face[1] + (self.margin * math.cos(-m))))

        # left 얼굴의 왼쪽
        left = (int(middle_of_face[0] - (self.margin * math.cos(m))),
                int(middle_of_face[1] - (self.margin * math.sin(m))))

        # right 얼굴의 오른쪽
        right = (int(middle_of_face[0] + (self.margin * math.cos(m))),
                 int(middle_of_face[1] + (self.margin * math.sin(m))))

        # left_upper 얼굴의 왼쪽 위
        left_upper = (int(top[0] - (self.margin * math.cos(m))),
                      int(top[1] - (self.margin * math.sin(m))))

        # left_lower 얼굴의 왼쪽 아래
        left_lower = (int(bottom[0] - (self.margin * math.cos(m))),
                      int(bottom[1] - (self.margin * math.sin(m))))

        # right_upper 얼굴의 오른쪽 위
        right_upper = (int(top[0] + (self.margin * math.cos(m))),
                       int(top[1] + (self.margin * math.sin(m))))

        # right_lower 얼굴의 오른쪽 아래
        right_lower = (int(bottom[0] + (self.margin * math.cos(m))),
                       int(bottom[1] + (self.margin * math.sin(m))))

        # original_patch 의 꼭짓점 반환
        min_xy = (min([x for x in [left_upper[0], left_lower[0], right_upper[0], right_lower[0]]]),
                  min([x for x in [left_upper[1], left_lower[1], right_upper[1], right_lower[1]]]))
        max_xy = (max([x for x in [left_upper[0], left_lower[0], right_upper[0], right_lower[0]]]),
                  max([x for x in [left_upper[1], left_lower[1], right_upper[1], right_lower[1]]]))

        # 각 꼭짓점 정의
        left_top = min_xy
        left_bottom = (min_xy[0], max_xy[1])
        right_top = (max_xy[0], min_xy[1])
        right_bottom = max_xy

        # test purpose
        output = self.original_image
        output = cv2.line(output, left_eye, left_eye, (255, 0, 0), 5)
        output = cv2.line(output, right_eye, right_eye, (0, 0, 255), 5)
        output = cv2.line(output, mouth, mouth, (0, 255, 0), 5)
        output = cv2.line(output, middle_of_face, middle_of_face, (128, 128, 128), 5)

        output = cv2.line(output, top, top, (255, 127, 127), 5)
        output = cv2.line(output, bottom, bottom, (127, 127, 255), 5)
        output = cv2.line(output, left, left, (127, 255, 127), 5)
        output = cv2.line(output, right, right, (0, 0, 0), 5)

        output = cv2.line(output, left_upper, left_lower, (0, 0, 255), 5)
        output = cv2.line(output, left_lower, right_lower, (0, 0, 255), 5)
        output = cv2.line(output, right_upper, left_upper, (0, 0, 255), 5)
        output = cv2.line(output, right_lower, right_upper, (0, 0, 255), 5)

        output = cv2.line(output, left_top, left_bottom, (0, 0, 0), 5)
        output = cv2.line(output, left_bottom, right_bottom, (0, 0, 0), 5)
        output = cv2.line(output, right_top, left_top, (0, 0, 0), 5)
        output = cv2.line(output, right_bottom, right_top, (0, 0, 0), 5)

        cv2.imwrite('test0.jpg', output)
        matrix = cv2.getRotationMatrix2D(middle_of_face, rotation, 1)
        output = cv2.warpAffine(output, matrix, (len(self.original_image[0]), len(self.original_image)))
        cv2.imwrite('test1.jpg', output)

        return left_top, right_top, left_bottom, right_bottom

    def get_origin_patch(self) -> np.ndarray:
        '''
        1920 * 1080 원본 이미지에서 origin_patch영역에 해당하는 이미지를 반환합니다.
        return : origin_patch영역에 해당하는 이미지
        '''
        pass

    def set_origin_patch(self, processed_image: np.ndarray) -> np.ndarray:
        '''
        1920 * 1080 원본 이미지의 origin_patch 영역을 processed_image로 대체해서 반환 합니다.
        param processed_image : 스타일 변환이 완료된 origin_patch
        return : 스타일 변환이 모두 완료된 1920 * 1080 크기의 이미지를 반환합니다.
        '''
        pass
