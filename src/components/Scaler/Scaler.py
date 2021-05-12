import numpy as np

class Scaler:
    def __init__(self, aligned_patch: np.ndarray, target_size: int = 512):
        '''
        
        '''
        self.target_size = target_size
        self.aligned_patch = aligned_patch
        self.origin_size = None
        pass

    def scale_forward(self) -> np.ndarray:
        '''
        target_size(512)로 이미지 크기 변환한 후 반환
        return : target_size x target_size 크기의 이미지
        '''
        pass

    def scale_backward(self) -> np.ndarray:
        '''
        원래 크기로 변환후 반환
        return : aligned_patch와 크기가 동일한 이미지
        '''
        pass