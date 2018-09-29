from PIL import Image
import os
import torchvision.utils
import numpy as np


def save_image(tensor, filename):
    if tensor.size()[0] == 1:
        tensor = tensor.cpu()[0, ...]
        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        im.save(filename)
    else:
        torchvision.utils.save_image(
            tensor, filename, normalize=False, range=(0, 1))
