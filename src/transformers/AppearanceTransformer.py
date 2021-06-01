import os

import cv2
import numpy as np

from src.components.Aligner.Aligner import Aligner
from src.components.BoundingBox.BoundingBox import BoundingBox
from src.components.MaskOrientGenerator.MaskOrientGenerator import MaskOrientGenerator
from src.components.Scaler.Scaler import Scaler
from src.transformers.Transformer import Transformer
from src.util.sender import Sender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class AppearanceTransformer(Transformer):
    def __init__(self):
        self.ref_image = 'a.jpg'
        self.sender = Sender()

    def set_reference(self):
        pass

    def transform(self, original_image: np.ndarray) -> np.ndarray:
        # original_image : 1920 x 1080
        # return : 1920 x 1080
        ref_image = cv2.imread(BASE_DIR + '/../../iu1.jpg')

        bounding_box_src = BoundingBox(original_image)
        bounding_box_ref = BoundingBox(ref_image)

        aligner_src = Aligner(bounding_box_src)
        aligner_ref = Aligner(bounding_box_ref)

        aligned_face_patch_src = aligner_src.align_forward()
        src_scaler = Scaler(aligned_face_patch_src)

        aligned_face_patch_ref = aligner_ref.align_forward()
        ref_scaler = Scaler(aligned_face_patch_ref)

        mask_orient = MaskOrientGenerator()
        scaled_src = src_scaler.scale_forward()
        mask, orient = mask_orient.generate(scaled_src)

        scaled_ref = ref_scaler.scale_forward()
        mask_ref, orient_ref = mask_orient.generate(scaled_ref)

        orient_mask = mask_orient.generate_mask(scaled_ref)

        generated_image: np.ndarray = self.sender.send_and_recv(datas={
            'label_ref': mask_ref,
            'label_tag': mask,
            'orient_mask': orient_mask,
            'orient_tag': orient,
            'orient_ref': orient_ref,
            'image_ref': scaled_ref,
            'image_tag': scaled_src,
        })

        generated_image = src_scaler.scale_backward(generated_image)
        generated_image = aligner_src.align_backward(generated_image)
        return bounding_box_src.set_origin_patch(generated_image)
