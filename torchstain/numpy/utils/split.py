import numpy as np
from torchstain.numpy.utils.rgb2lab import rgb2lab


def csplit(I):
    return np.dsplit(I, I.shape[-1])


def cmerge(I1, I2, I3):
    return np.dstack((I1, I2, I3))


def lab_split(I):
    I = rgb2lab(I)
    I1, I2, I3 = csplit(I)
    return I1 / 2.55, I2 - 128, I3 - 128


def lab_merge(I1, I2, I3):
    merged = cmerge((I1 * 2.55, I2 + 128, I3 + 128))
    return np.clip(merged, 0, 255)
