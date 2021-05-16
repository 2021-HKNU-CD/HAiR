import cv2

from src.components.BoundingBox.BoundingBox import BoundingBox

BB = BoundingBox(original_image=cv2.imread('data/test/unnamed.jpg'),
                 facefeat_model_path='models/checkpoints/shape_predictor_68_face_landmarks.dat')

BB.get_bounding_box()
