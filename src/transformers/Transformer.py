from abc import *
import numpy as np

class Transformer(metaclass=ABCMeta):
    @abstractmethod
    def set_reference(self):
        pass
    
    @abstractmethod
    def transform(self, original_image: np.ndarray) -> np.ndarray:
        # original_image : 1920 x 1080
        # return : 1920 x 1080
        pass