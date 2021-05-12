import numpy as np
from src.transformers.Transformer import Transformer

class ShapeStructureTransformer(Transformer):
    def __init__(self):
        self.C = 10
        self.ref = 'a.jpg'
    def set_reference(self):
        pass
    def transform(self, original_image: np.ndarray) -> np.ndarray:
        # original_image : 1920 x 1080
        # return : 1920 x 1080
        pass