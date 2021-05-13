import numpy as np

from calOrient import Orient
from faceSegmentation import FaceSegmentation


class MaskOrientGenerator:
    def __init__(self):
        self.faceSegmentation = FaceSegmentation()
        self.orient = Orient()

    def generate(self, aligned_scaled_patch: np.ndarray) -> tuple:
        '''
        512 * 512 크기의 aligned_scaled_patch로 부터 마스크와 오리엔트 덴스를 반환합니다.
        param aligned_scaled_patch : align과 scale이 수행된 512 * 512 크기의 이미지
        return   mask : aligned_scaled_patch의 마스크 이미지
               orient : aligned_scaled_patch의 오리엔트 덴스
        '''
        mask = self.generate_mask(aligned_scaled_patch)
        orient = self.generate_orient(aligned_scaled_patch, mask)
        return mask, orient

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        '''
        512 * 512 원본 이미지로 부터 마스크 이미지를 반환합니다.
        param image : 원본 이미지
        return : image의 마스크
        '''
        return self.faceSegmentation.image_to_mask(image, 256, 512)

    def generate_orient(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        '''
        512 * 512 마스크 이미지로 부터 오리엔트 덴스를 반환합니다.
        param image : 원본 이미지
              mask  : 원본 이미지의 마스크
        return : image의 orient dense
        '''
        return self.orient.makeOrient(image, mask)
