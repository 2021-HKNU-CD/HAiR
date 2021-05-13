import numpy as np

from models.faceFeature import faceFeature


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
        self.theta = None
        self.face_center = None
        self.faceFeat = faceFeature()

    def get_bounding_box(self) -> tuple[int, int, int, int]:
        '''
        1920 * 1080 원본 이미지에서 origin_patch영역의 좌상단, 우상단, 좌하단, 우하단 영역을 반환합니다.
        '''

        # BoundingBox = 얼굴 중심 ~ Margin*(sin(회전각) + cos(회전각))
        left_top, right_top, left_bottom, right_bottom = 0, 0, 0, 0

        left_eye, right_eye, mouth = self.faceFeat.get(self.original_image)



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