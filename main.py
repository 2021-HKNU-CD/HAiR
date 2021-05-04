import numpy as np

from faceSegmentation import FaceSegmentation, img_to_ndarray
from calOrient import Orient, save_to_img

FS = FaceSegmentation('checkpoints/model.pt')
mask = FS.image_to_mask(img_to_ndarray('99999.jpg'), 256, 512)
orient = Orient()
orient_dense = orient.makeOrient(img_to_ndarray('99999.jpg'), mask)
save_to_img(orient_dense, "99999")
