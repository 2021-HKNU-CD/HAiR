import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def save_to_img(orient_np: np.ndarray, filename: str) -> None:
    Image.fromarray(orient_np).save(f"./{filename}.png")


class Orient(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, stride=1, padding=8):
        super(Orient, self).__init__()
        self.criterion = nn.L1Loss()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.filter = self.DoG_fn

        self.numKernels = 32
        self.kernel_size = 17

    def DoG_fn(self, kernel_size, channel_in, channel_out, theta):
        # params
        sigma_h = nn.Parameter(torch.ones(channel_out) * 1.0, requires_grad=False)
        sigma_l = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False)
        sigma_y = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False)

        # Bounding box
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1)
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
        x_0 = torch.arange(xmin, xmax + 1)
        # [channel_out, channelin, kernel, kernel]
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()

        # Rotation
        # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + \
                  y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + \
                  y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = (torch.exp(-.5 * (x_theta ** 2 / sigma_h.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1,
                                                                                                           1) ** 2)) / sigma_h
              - torch.exp(-.5 * (x_theta ** 2 / sigma_l.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1,
                                                                                                             1) ** 2)) / sigma_l) / (
                     1.0 / sigma_h - 1.0 / sigma_l)

        return gb

    def calOrientation(self, image, mask=None):
        resArray = []
        # filter the image with different orientations
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(
                self.channel_out) * (math.pi * iOrient / self.numKernels), requires_grad=False)
            filterKernel = self.filter(
                self.kernel_size, self.channel_in, self.channel_out, theta)
            filterKernel = filterKernel.float()
            response = F.conv2d(image, filterKernel,
                                stride=self.stride, padding=self.padding)
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(
            resTensor, dim=1).float()  # range from 0 to 31
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1)

        return maxResTensor, confidenceTensor

    def makeOrient(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        trans_image = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = trans_image(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        fake_image = (image_tensor + 1) / 2.0 * 255
        gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
        gray = torch.unsqueeze(gray, 1)
        orient_tensor, confidence_tensor = self.calOrientation(gray)

        orient_tensor = orient_tensor * math.pi / 31 * 2
        mask_tensor = torch.from_numpy(mask).float()
        flow_x = torch.cos(orient_tensor) * confidence_tensor * mask_tensor
        flow_y = torch.sin(orient_tensor) * confidence_tensor * mask_tensor
        flow_x = torch.from_numpy(cv2.GaussianBlur(flow_x.numpy().squeeze(), (0, 0), 4))
        flow_y = torch.from_numpy(cv2.GaussianBlur(flow_y.numpy().squeeze(), (0, 0), 4))
        orient_tensor = torch.atan2(flow_y, flow_x) * 0.5
        orient_tensor[orient_tensor < 0] += math.pi
        orient_np = orient_tensor.numpy().squeeze() * 255. / math.pi * mask
        return np.uint8(orient_np)
