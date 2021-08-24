import cv2
import numpy as np

from src.components.BoundingBox.BoundingBox import BoundingBox
from .wing import FaceAligner, align_face, align_face_restore
from torchvision.utils import make_grid
from torch import uint8
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class AlignerWing:
    """
    origin_patch와 얼굴중심, margin, 회전각 theta를 받아

    align_forwar시
    회전하고 잘라서 만든 align_patch를 반환

    aling_backward시
    헤어스타일이 변경된 align_patch를 받으면 원래 origin_patch에 덮어쓴 origin_patch를 반환
    """
    FaceAligner = FaceAligner(os.path.join(BASE_DIR, '../../../models/checkpoints/wing/wing.ckpt'),
                                   os.path.join(BASE_DIR, '../../../models/checkpoints/wing/celeba_lm_mean.npz'),
                                   512)

    def __init__(self, bounding_box: BoundingBox = None):
        if bounding_box is not None:
            self.set_bounding_box(bounding_box)

        self.origin_landmarks = None

    def set_bounding_box(self, bounding_box: BoundingBox):
        self.bounding_box = bounding_box
        self.bounding_box.get_bounding_box()
        self.origin_patch = self.bounding_box.get_origin_patch()

    def check_initialized(self):
        if self.bounding_box is None:
            raise Exception("bounding_box is None")

    def align_forward(self) -> np.ndarray:
        '''
        origin_patch로 부터 aligned_patch를 반환합니다.
        return : margin * margin 크기의 aligned_patch
        '''
        self.check_initialized()

        def denormalize(x):
            out = (x + 1) / 2
            return out.clamp_(0, 1)

        result, origin_landmarks = align_face(512, self.origin_patch, AlignerWing.FaceAligner)
        self.origin_landmarks = origin_landmarks
        result = denormalize(result)
        # result : tensor

        grid = make_grid(result)
        result = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', uint8).numpy()
        # result : ndarray
        return result

    def align_backward(self, style_changed_aligned_patch: np.ndarray) -> np.ndarray:
        '''
        스타일 변환이 완료된 style_changed_aligned_patch로 부터 style_changed_origin_patch를 반환합니다.
        param style_changed_aligned_patch : 기존 aligned_patch에서 스타일만 변경된 이미지
        return : 기존 origin_patch에서 스타일만 변경된 이미지
        '''
        self.check_initialized()

        def denormalize(x):
            out = (x + 1) / 2
            return out.clamp_(0, 1)

        if self.origin_landmarks is None:
            raise Exception("origin_landmakrs가 없습니다. align_forward를 먼저 실행해 주세요.")

        result = align_face_restore(512, style_changed_aligned_patch, self.origin_landmarks, AlignerWing.FaceAligner)
        result = denormalize(result)
        # result : tensor

        grid = make_grid(result)
        result = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', uint8).numpy()
        # result : ndarray

        H, W, _ = self.origin_patch.shape
        result = cv2.resize(result, (H*2, W*2), interpolation=cv2.INTER_LINEAR)

        _, threshold = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
        threshold = threshold.astype(np.uint8)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours[0])):
            y, x = contours[0][i][0]
            y_ = ((H*2)//2 + y) // 2
            x_ = ((W*2)//2 + x) // 2
            contours[0][i][0] = [y_, x_]

        _mask = np.ones_like(result) * 0
        points = np.array(contours[0], np.int32)
        color = (255, 255, 255)
        cv2.fillPoly(_mask, [points], color)

        mask = np.zeros(result.shape[:2], np.uint8)
        for i in range(len(_mask)):
            row = _mask[i]
            for j in range(len(row)):
                mask[i][j] = _mask[i][j][0]

        ret = np.zeros(self.origin_patch.shape, np.uint8)
        for i, row in enumerate(ret):
            for j, col in enumerate(row):
                ret[i][j] = result[(H//2)+i][(W//2)+j] if mask[(H//2)+i][(W//2)+j] == 255 else self.origin_patch[i][j]

        return ret

# Test
def test(file):
    img = cv2.imread(file + '.jpg')
    bBox = BoundingBox(img)
    aw = AlignerWing(bBox)
    aligned = aw.align_forward()
    cv2.imwrite(file + '_bb_2.jpg', bBox.get_origin_patch())
    cv2.imwrite(file + '_forward_2.jpg', aligned)
    cv2.imwrite(file + '_backward_2.jpg', aw.align_backward(aligned))