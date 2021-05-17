import cv2

from src.components.BoundingBox.BoundingBox import BoundingBox

BB = BoundingBox(original_image=cv2.imread('data/test/iu2.jpg'),
                 facefeat_model_path='models/checkpoints/shape_predictor_68_face_landmarks.dat')

BB.get_bounding_box()
cv2.imwrite('test2.jpg', BB.get_origin_patch())
cv2.imwrite('test3.jpg', BB.set_origin_patch(cv2.imread('processed.jpg')))