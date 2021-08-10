import numpy as np

import cv2

from models.baldgan import BaldGAN
from src.components.Aligner.AlignerWing import AlignerWing
from src.components.BoundingBox.BoundingBox import BoundingBox
from src.components.MaskOrientGenerator.MaskOrientGenerator import MaskOrientGenerator
from src.components.Scaler.Scaler import Scaler
from src.util.sender import Sender

class Transformer:
    def __init__(self):

        self.appearance_ref = None
        self.shape_ref = None
        self.structure_ref = None

        self.sender = Sender()
        self.MOGenerator = MaskOrientGenerator()

    def set_appearance_ref(self, ref: np.ndarray):
        self.appearance_ref = ref

    def set_shape_ref(self, ref: np.ndarray):
        self.shape_ref = ref

    def set_structure_ref(self, ref: np.ndarray):
        self.structure_ref = ref

    def transform(self, original_image: np.ndarray) -> np.ndarray:
        # original_image : 1920 x 1080
        # return : 1920 x 1080

        boundingBox = BoundingBox(original_image)
        aligner = AlignerWing(boundingBox)

        balder = Balder(BaldGAN())
        src = self._src_preprocess(aligner, balder)

        appearance_ref, shape_ref, structure_ref = None, None, None
        if self.appearance_ref is not None:
            appearance_ref = self._ref_preprocess(AlignerWing(BoundingBox(self.appearance_ref)))

        if self.shape_ref is not None:
            shape_ref = self._ref_preprocess(AlignerWing(BoundingBox(self.shape_ref)))

        if self.structure_ref is not None:
            structure_ref = self._ref_preprocess(AlignerWing(BoundingBox(self.structure_ref)))

        # Appearance
        appearance_mask = src['mask'] if appearance_ref is None else appearance_ref['mask']
        appearance_img = src['img_origin'] if appearance_ref is None else appearance_ref['img']

        # Shape
        shape_mask = src['mask'] if shape_ref is None else shape_ref['mask']

        # Structure
        orient = src['orient'] if structure_ref is None else structure_ref['orient']

        # App : appearance_ref
        # Shape : shape_ref
        # Structure : structure_ref
        # None인 항목은 src의 속성을 유지하도록 변환
        datas = {
            'label_ref': appearance_mask,  # appearance
            'label_tag': shape_mask, # shape

            'orient_mask': shape_mask,  # structure ref mask
            'orient_tag': orient,  # src orient ???????
            'orient_ref': orient,  # structure ref orient

            'image_ref': appearance_img,  # appearance
            'image_tag': src['img_bald'],  # src
        }
        generated: np.ndarray = self.sender.send_and_recv(datas)

        generated = balder.scaler.scale_backward(generated)
        generated = aligner.align_backward(generated)
        return boundingBox.set_origin_patch(generated)

    def _ref_preprocess(self, aligner):
        scaled_ref = Scaler((aligner.align_forward())).scale_forward()
        mask_ref, orient_ref = self.MOGenerator.generate(scaled_ref)
        ret = {
            'img': scaled_ref,
            'mask': mask_ref,
            'orient': orient_ref
        }
        return ret

    def _src_preprocess(self, aligner, bald):
        aligned_face_patch_src = aligner.align_forward()

        # bald
        balded_src = bald.run(aligned_face_patch_src)

        src_origin = Scaler(aligned_face_patch_src).scale_forward()
        mask_src, orient_src = self.MOGenerator.generate(src_origin)

        ret = {
            'img_origin': src_origin,
            'img_bald': balded_src,
            'mask': mask_src,
            'orient': orient_src
        }
        return ret

class Balder:
    def __init__(self, baldGAN):
        self.baldGAN = baldGAN
        self.scaler = None

    def run(self, aligned_face_patch_src):
        self.scaler = Scaler(aligned_face_patch_src, target_size=256)
        scaled_256_src = self.scaler.scale_forward()
        self.scaler = Scaler(self.baldGAN.go_bald(scaled_256_src))
        return self.scaler.scale_forward()

class NoBalder:
    def __init__(self):
        self.scaler = None

    def run(self, aligned_face_patch_src):
        self.scaler = Scaler(aligned_face_patch_src)
        return self.scaler.scale_forward()

def getTransformer() -> Transformer:
    return Transformer()

def test():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    transformer = getTransformer()

    appearance_refs = {
        'none' : None,
        'lee' : cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, 'lee'))
    }

    structure_refs = {
        'none' : None,
        '10' : cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, '10')),
        'kim': cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, 'kim'))
    }

    shape_refs = {
        'none' : None,
        '10': cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, '10')),
        'kim' : cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, 'kim'))
    }

    for img_idx in [2]:
        src = cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, img_idx))
        for a_key in appearance_refs.keys():
            appearance_ref = appearance_refs[a_key]

            for st_key in structure_refs.keys():
                structure_ref = structure_refs[st_key]

                for sh_key in shape_refs.keys():
                    shape_ref = shape_refs[sh_key]

                    file = 'results/{}_{}_{}_{}.jpg'.format(img_idx, a_key, st_key, sh_key)
                    print(file, 'started')

                    transformer.set_appearance_ref(appearance_ref)
                    transformer.set_shape_ref(shape_ref)
                    transformer.set_structure_ref(structure_ref)

                    try:
                        result = transformer.transform(src)
                        cv2.imwrite(file, result)
                    except:
                        print('error')

# test()