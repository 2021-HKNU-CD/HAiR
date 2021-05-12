import numpy as np

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
        self.origin_patch = self.bounding_box.get_origin_patch()
        self.rotated_patch = None
        self.aligned_patch = None
        pass

    def align_forward(self) -> np.ndarray:
        '''
        origin_patch로 부터 aligned_patch를 반환합니다.
        return : margin * margin 크기의 aligned_patch
        '''
        origin_patch = self.bounding_box.get_bounding_box()

        # 1. origin_patch를 theta만큼 회전
        # 2. 얼굴 중심 기준 margin만큼 잘라내기

        aligned_patch = None
        return aligned_patch

    def align_backward(self, style_changed_aligned_patch: np.ndarray) -> np.ndarray:
        '''
        스타일 변환이 완료된 style_changed_aligned_patch로 부터 style_changed_origin_patch를 반환합니다.
        param style_changed_aligned_patch : 기존 aligned_patch에서 스타일만 변경된 이미지
        return : 기존 origin_patch에서 스타일만 변경된 이미지
        '''
        # 1. rotated_patch에 덮어쓰기
        # 2. -회전각 만큼 회전
        # 3. origin_patch에 덮어쓰기
        style_changed_origin_patch = None
        return style_changed_origin_patch