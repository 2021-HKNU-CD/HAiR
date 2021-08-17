from src.components.Aligner.AlignerWing import AlignerWing
from src.components.BoundingBox.BoundingBox import BoundingBox
from models.baldgan import BaldGAN, model_path
from src.components.Scaler.Scaler import Scaler

class Balder:
    def __init__(self, baldGAN):
        self.baldGAN = baldGAN
        self.scaler = None

    def run(self, aligned_face_patch_src):
        self.scaler = Scaler(aligned_face_patch_src, target_size=256)
        scaled_256_src = self.scaler.scale_forward()
        return Scaler(self.baldGAN.go_bald(scaled_256_src)).scale_forward()

class NoBalder:
    def __init__(self):
        self.scaler = None

    def run(self, aligned_face_patch_src):
        self.scaler = Scaler(aligned_face_patch_src)
        return self.scaler.scale_forward()

def BoundingBoxFactory(img):
    return BoundingBox(img)

def AlignerWingFactory(boundingBox):
    return AlignerWing(boundingBox)

def AlignerFactory(boundingBox):
    from src.components.Aligner.Aligner import Aligner
    return Aligner(boundingBox)

def BalderFactory():
    return Balder(baldGans['model_G_5_170'])

def Balder_5_170_Factory():
    return Balder(baldGans['model_G_5_170_retrained'])

def Balder_10_170_Factory():
    return Balder(baldGans['model_G_10_340'])

check_points = ['model_G_5_170', 'model_G_5_170_retrained', 'model_G_10_340']

def b(ckpt):
    bald = BaldGAN()
    bald.model.load_weights(model_path + ckpt + '.hdf5')
    return bald

baldGans = {ckpt: b(ckpt) for ckpt in check_points}