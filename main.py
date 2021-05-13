from src.components.MaskOrientGenerator.faceSegmentation import FaceSegmentation, img_to_ndarray
from src.components.MaskOrientGenerator.calOrient import Orient, save_to_img

FS = FaceSegmentation('checkpoints/model.pt')
mask = FS.image_to_mask(img_to_ndarray('data/test/99999.jpg'), 256, 512)
orient = Orient()
orient_dense = orient.makeOrient(img_to_ndarray('data/test/99999.jpg'), mask)
save_to_img(orient_dense, "99999")
