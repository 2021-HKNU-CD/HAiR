from src.transformers.AppearanceTransformer import AppearanceTransformer 
from src.transformers.ShapeStructureTransformer import ShapeStructureTransformer

transformers = {'appreance' : AppearanceTransformer(), 'shape' : ShapeStructureTransformer()}
mode = 'appearance'

CameraInput = None
transformers[mode].transform(CameraInput)