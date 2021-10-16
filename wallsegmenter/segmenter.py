import os

import numpy as np
import PIL
import torch
import torchvision.transforms

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225],
        ),  # across a large photo dataset.
    ]
)
# Function for segmenting wall in the input image
def segment_image(segmentation_module, img, device):
    img_data = transform(img)
    singleton_batch = {"img_data": img_data[None].to(device)}
    seg_size = np.array(img).shape[:2]

    with torch.no_grad():
        scores = segmentation_module(singleton_batch, seg_size=seg_size)

    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred


def get_mask(img, pred):
    img_green = img.copy()
    black_green = img.copy()
    img_green[pred == 0] = [0, 255, 0]
    black_green[pred == 0] = [0, 255, 0]
    black_green[pred != 0] = [0, 0, 0]
    return black_green, img_green


def visualize_wall(img, pred):
    black_green, img_green = get_mask(img, pred)
    im_vis = np.concatenate((img, black_green, img_green), axis=1)
    return PIL.Image.fromarray(im_vis)
