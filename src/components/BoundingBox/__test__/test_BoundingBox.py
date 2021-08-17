import unittest

import cv2

from src.components.BoundingBox.BoundingBox import BoundingBox


class TestBoundingBox(unittest.TestCase):
    def test_BoundingBox(self):
        bb = BoundingBox(cv2.imread('../../../../test_images/testface_1.jpg'))
        self.assertIsNotNone(bb.get_bounding_box())

    def test_assert_off_axis_left_upper(self):
        bb = BoundingBox(cv2.imread('../../../../test_images/testface_1_offaxis_left_upper.jpg'))
        with self.assertRaisesRegex(ValueError, "BoundingBoxOffLimitError"):
            bb.get_bounding_box()

    def test_assert_off_axis_left_lower(self):
        bb = BoundingBox(cv2.imread('../../../../test_images/testface_1_offaxis_left_lower.jpg'))
        with self.assertRaisesRegex(ValueError, "BoundingBoxOffLimitError"):
            bb.get_bounding_box()

    def test_assert_off_axis_right_upper(self):
        bb = BoundingBox(cv2.imread('../../../../test_images/testface_1_offaxis_right_upper.jpg'))
        with self.assertRaisesRegex(ValueError, "BoundingBoxOffLimitError"):
            bb.get_bounding_box()

    def test_assert_off_axis_right_lower(self):
        bb = BoundingBox(cv2.imread('../../../../test_images/testface_1_offaxis_right_lower.jpg'))
        with self.assertRaisesRegex(ValueError, "BoundingBoxOffLimitError"):
            bb.get_bounding_box()

    def test_assert_no_face(self):
        bb = BoundingBox(cv2.imread('../../../../test_images/testface_1_noface.jpg'))
        with self.assertRaisesRegex(ValueError, "NoFaceError"):
            bb.get_bounding_box()




if __name__ == '__main__':
    unittest.main()
