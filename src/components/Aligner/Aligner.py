import cv2
import numpy as np
from numpy.lib import math

from src.components.BoundingBox import BoundingBox


class Aligner:
    """
    origin_patch와 얼굴중심, margin, 회전각 theta를 받아

    align_forwar시 
    회전하고 잘라서 만든 align_patch를 반환
    
    aling_backward시
    헤어스타일이 변경된 align_patch를 받으면 원래 origin_patch에 덮어쓴 origin_patch를 반환
    """

    def __init__(self, bounding_box: BoundingBox):
        self.bounding_box = bounding_box
        self.bounding_box.get_bounding_box()
        self.origin_patch = self.bounding_box.get_origin_patch()
        self.rotated_patch = None
        self.aligned_patch = None

    def align_forward(self) -> np.ndarray:
        '''
        origin_patch로 부터 aligned_patch를 반환합니다.
        return : margin * margin 크기의 aligned_patch
        '''
        origin_patch: np.ndarray = self.bounding_box.get_origin_patch()
        face_center: tuple[int, int] = (len(self.origin_patch[0]) // 2, len(self.origin_patch) // 2)
        margin: float = self.bounding_box.margin
        rotation: float = self.bounding_box.rotation

        # 1. origin_patch를 theta만큼 회전
        matrix = cv2.getRotationMatrix2D(face_center, rotation, 1 / math.sqrt(2))
        margin = int((1 / math.sqrt(2)) * margin)
        aligned_patch: np.ndarray = cv2.warpAffine(origin_patch, matrix, (len(origin_patch[0]), len(origin_patch)))
        self.rotated_patch = aligned_patch.copy()
        # 2. 얼굴 중심 기준 margin만큼 잘라내기
        aligned_patch = aligned_patch[
                        face_center[0] - margin:face_center[0] + margin,
                        face_center[1] - margin:face_center[1] + margin
                        ]

        return aligned_patch

    def align_backward(self, style_changed_aligned_patch: np.ndarray) -> np.ndarray:
        '''
        스타일 변환이 완료된 style_changed_aligned_patch로 부터 style_changed_origin_patch를 반환합니다.
        param style_changed_aligned_patch : 기존 aligned_patch에서 스타일만 변경된 이미지
        return : 기존 origin_patch에서 스타일만 변경된 이미지
        '''
        face_center: tuple[int, int] = (len(self.origin_patch[0]) // 2, len(self.origin_patch) // 2)
        margin: float = self.bounding_box.margin
        rotation: float = self.bounding_box.rotation
        margin = int((1 / math.sqrt(2)) * margin)
        # 1. rotated_patch에 덮어쓰기
        modified_origin_patch = self.rotated_patch.copy()
        modified_origin_patch[
        face_center[0] - margin: face_center[0] + margin,
        face_center[1] - margin: face_center[1] + margin
        ] = style_changed_aligned_patch
        # 2. -회전각 만큼 회전
        matrix_rev = cv2.getRotationMatrix2D(face_center, -rotation, math.sqrt(2))
        modified_origin_patch = cv2.warpAffine(modified_origin_patch, matrix_rev, (
            len(modified_origin_patch[0]), len(modified_origin_patch)))
        # 3. origin_patch에 덮어쓰기
        return modified_origin_patch
