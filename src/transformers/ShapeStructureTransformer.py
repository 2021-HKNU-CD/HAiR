import os

import cv2
import numpy as np

from models.baldgan import BaldGAN
from src.components.Aligner.Aligner import Aligner
from src.components.BoundingBox.BoundingBox import BoundingBox
from src.components.MaskOrientGenerator.MaskOrientGenerator import MaskOrientGenerator
from src.components.Scaler.Scaler import Scaler
from src.transformers.Transformer import Transformer
from src.util.sender import Sender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ShapeStructureTransformer(Transformer):
    def __init__(self):
        self.C = 10
        self.shape_ref = 'shape_ref.jpg'
        self.structure_ref = 'structure_ref.jpg'
        self.sender = Sender()

    def set_reference(self):
        pass

    def transform(self, original_image: np.ndarray) -> np.ndarray:
        # original_image : 1920 x 1080
        # return : 1920 x 1080
        MOGenerator = MaskOrientGenerator()

        shape_ref_image = cv2.imread(BASE_DIR + '/../../data/' + self.shape_ref)
        structure_ref_image = cv2.imread(BASE_DIR + '/../../data/' + self.structure_ref)

        # src
        bounding_box_src = BoundingBox(original_image)
        aligner_src = Aligner(bounding_box_src)
        aligned_face_patch_src = aligner_src.align_forward()
        src_to_baldgan_scaler = Scaler(aligned_face_patch_src, target_size=256)
        scaled_256_src = src_to_baldgan_scaler.scale_forward()

        balded_src = Scaler(BaldGAN().go_bald(scaled_256_src)).scale_forward()
        mask_src, orient_src = MOGenerator.generate(balded_src)

        # shape ref
        scaled_shape_ref = Scaler(Aligner(BoundingBox(shape_ref_image)).align_forward()).scale_forward()
        cv2.imwrite('scaled_shape_ref.jpg', scaled_shape_ref)
        mask_shape_ref, orient_shape_ref = MOGenerator.generate(scaled_shape_ref)

        # structure ref
        scaled_structure_ref = Scaler(Aligner(BoundingBox(structure_ref_image)).align_forward()).scale_forward()
        cv2.imwrite('scaled_structure_ref.jpg', scaled_structure_ref)
        mask_structure_ref, orient_structure_ref = MOGenerator.generate(scaled_structure_ref)

        datas = {
            'label_ref': mask_structure_ref,
            'label_tag': mask_src,

            'orient_mask': mask_structure_ref,
            'orient_tag': mask_shape_ref,
            'orient_ref': orient_structure_ref,

            'image_ref': structure_ref_image,
            'image_tag': balded_src,
        }

        generated_image: np.ndarray = self.sender.send_and_recv(datas)
        generated_image = src_to_baldgan_scaler.scale_backward(generated_image)
        generated_image = aligner_src.align_backward(generated_image)
        return bounding_box_src.set_origin_patch(generated_image)

# Test code
# cv2.imwrite('2hochang_1.jpg', ShapeStructureTransformer().transform(cv2.imread(BASE_DIR + '/../../data/2hochang_2.jpg')))
