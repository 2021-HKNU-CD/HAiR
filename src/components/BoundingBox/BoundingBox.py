import numpy as np
from numpy.lib import math

from models.faceFeature import FaceFeature


class BoundingBox:
    """
    origin_image로 부터 눈, 코, 입 좌표를 추출하고
    얼굴 중심을 구한 뒤 회전각을 구하고 Margin을 구해서 origin_image에서 bounding box를 구합니다.
    """

    def __init__(self, original_image: np.ndarray):
        '''
        param original_image : 1920 * 1080 크기의 카메라로 들어온 입력 이미지
        '''
        self.original_image = original_image
        self.margin = None
        self.rotation = None
        self.theta = None
        self.face_center = None
        self.faceFeat = FaceFeature()
        self.image_coords = []

    def get_bounding_box(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        1920 * 1080 원본 이미지에서 origin_patch 영역의 좌상단, 우상단, 좌하단, 우하단 영역을 반환합니다.
        """

        # BoundingBox = 얼굴 중심 ~ Margin*(sin(회전각) + cos(회전각))

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
        self.theta = m
        self.rotation = rotation

        # Margin
        distance_between_eyes = abs(y1 - y2) + abs(x1 - x2)
        self.margin = distance_between_eyes * 2

        # top 얼굴의 위
        top = (int(middle_of_face[0] - (self.margin * math.sin(-m))),
               int(middle_of_face[1] - (self.margin * math.cos(-m))))

        # bottom 얼굴의 아래
        bottom = (int(middle_of_face[0] + (self.margin * math.sin(-m))),
                  int(middle_of_face[1] + (self.margin * math.cos(-m))))

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
        self.image_coords = [left_top, right_top, left_bottom, right_bottom]
        return left_top, right_top, left_bottom, right_bottom

    def get_origin_patch(self) -> np.ndarray:
        """
        1920 * 1080 원본 이미지에서 origin_patch영역에 해당하는 이미지를 반환합니다.
        return : origin_patch영역에 해당하는 이미지
        """
        left_top, right_top, left_bottom, right_bottom = self.image_coords
        return self.original_image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

    def set_origin_patch(self, processed_image: np.ndarray) -> np.ndarray:
        """
        1920 * 1080 원본 이미지의 origin_patch 영역을 processed_image로 대체해서 반환 합니다.
        param processed_image : 스타일 변환이 완료된 origin_patch
        return : 스타일 변환이 모두 완료된 1920 * 1080 크기의 이미지를 반환합니다.
        """
        left_top, right_top, left_bottom, right_bottom = self.image_coords
        origin: np.ndarray = self.original_image.copy()
        origin[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] = processed_image
        return origin
