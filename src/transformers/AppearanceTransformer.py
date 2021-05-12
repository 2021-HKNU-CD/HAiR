import numpy as np
from src.transformers.Transformer import Transformer
from src.components.Aligner import Aligner
from src.components.BoundingBox import BoundingBox
from src.components.Scaler import Scaler
from src.components.MaskOrientGenerator import MaskOrientGenerator

class AppearanceTransformer(Transformer):
    def __init__(self):
        self.C = 10
        self.ref = 'a.jpg'
    def set_reference(self):
        pass
    def transform(self, original_image: np.ndarray) -> np.ndarray:
        # original_image : 1920 x 1080
        # return : 1920 x 1080
        MichiGAN = None
        ref_image = None
        
        bounding_box_src = BoundingBox(original_image)
        bounding_box_ref = BoundingBox(ref_image)

        aligner_src = Aligner(bounding_box_src)
        aligner_ref = Aligner(bounding_box_ref)
        
        aligned_face_patch_src = aligner_src.align_forward()
        src_scaler = Scaler(aligned_face_patch_src)
        
        aligned_face_patch_ref = aligner_ref.align_forward(ref_image)
        ref_scaler = Scaler(aligned_face_patch_ref)

        scaled_src = src_scaler.scale_forward()
        mask, orient = MaskOrientGenerator.generate(scaled_src)

        scaled_ref = ref_scaler.scale_forward()
        mask_ref, orient_ref = MaskOrientGenerator(scaled_ref)
        
        generated_image = MichiGAN.forward(scaled_src, mask, orient, scaled_ref, mask_ref, orient_ref)

        generated_image = src_scaler.scale_backward(generated_image)
        generated_image = aligner_src.align_backward(generated_image)
        bounding_box_src.set_origin_patch(generated_image)
        return None