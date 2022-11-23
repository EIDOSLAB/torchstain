import numpy as np
from torchstain.numpy.utils.rgb2lab import _rgb2xyz
import cv2 as cv
import skimage


_xyz2rgb = np.linalg.inv(_rgb2xyz)


def lab2rgbY(lab):
    return cv.cvtColor(np.clip(lab, 0, 255).astype("uint8"), cv.COLOR_LAB2RGB)


def lab2rgb(lab):
    lab[..., 0] /= 2.55
    lab[..., 1] -= 128
    lab[..., 2] -= 128
    return skimage.color.lab2rgb(lab)
