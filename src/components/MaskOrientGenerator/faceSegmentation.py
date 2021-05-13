import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from models.MobileNetV2_unet import MobileNetV2_unet


def write_to_txt(mask_arr: np.ndarray, mask_name: str) -> None:
    with open(f'{mask_name}.txt', 'wt') as opt_file:
        for i in mask_arr:
            for j in i:
                opt_file.write(str(j))
            opt_file.write('\n')


def img_to_ndarray(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class FaceSegmentation():
    """
    mostly come from
    https://github.com/kampta/face-seg
    """

    def __init__(self):
        model = MobileNetV2_unet(None).to(torch.device("cpu"))
        model_path = '../../../models/checkpoints/model.pt'
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model '{model_path}' is loaded")
        self.model = model

    def image_to_mask(self, image: np.ndarray, mask_size: int, out_size: int) -> np.ndarray:

        transform = transforms.Compose([
            transforms.Resize((mask_size, mask_size)),
            transforms.ToTensor(), ])

        pil_img = Image.fromarray(image)
        torch_img = transform(pil_img)
        torch_img = torch_img.unsqueeze(0)
        torch_img = torch_img.to(torch.device("cpu"))

        # feed-fowarding
        logits = self.model(torch_img)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)

        # 이 모델은 현재 얼굴을 1로 머리는 2로 처리하기때문에 머리만 1로 처리하는 과정임.
        mask = mask.squeeze()
        mask = np.where(mask == 1, 0, mask)
        mask = np.where(mask == 2, 1, mask)
        mask = np.array(mask, dtype='uint8')

        mask = cv2.resize(mask, dsize=(0, 0), fx=(out_size/mask_size),
                          fy=(out_size/mask_size), interpolation=cv2.INTER_LINEAR)

        return mask