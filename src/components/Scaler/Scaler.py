import numpy as np
import cv2

class Scaler:
    def __init__(self, aligned_patch: np.ndarray, target_size: int = 512):
        '''
        
        '''
        self.target_size = target_size
        self.aligned_patch = aligned_patch
        if aligned_patch.shape[0] != aligned_patch.shape[1]:
            raise Exception('aligned_patch의 가로 세로 크기가 동일하지 않습니다.')

        self.origin_size = aligned_patch.shape[0]
        pass

    def scale_forward(self) -> np.ndarray:
        '''
        target_size(512)로 이미지 크기 변환한 후 반환
        return : target_size x target_size 크기의 이미지
        '''
        return cv2.resize(self.aligned_patch,
                          dsize=(self.target_size, self.target_size),
                          interpolation=cv2.INTER_LINEAR)

    def scale_backward(self, style_changed_512) -> np.ndarray:
        '''
        원래 크기로 변환후 반환
        param style_changed_512 : MichiGAN을 통과해서 나온 512 x 512 이미지
        return : aligned_patch와 크기가 동일한 이미지
        '''
        return cv2.resize(style_changed_512,
                          dsize=(self.origin_size, self.origin_size),
                          interpolation=cv2.INTER_LINEAR)