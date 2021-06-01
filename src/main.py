from src.transformers.AppearanceTransformer import AppearanceTransformer 
from src.transformers.ShapeStructureTransformer import ShapeStructureTransformer

transformers = {
    'appearance': AppearanceTransformer(),
    'shape': ShapeStructureTransformer()
}
mode = 'appearance'

CameraInput = None
transformers[mode].transform(CameraInput)