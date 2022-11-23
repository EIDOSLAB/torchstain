import numpy as np
from torchstain.numpy.utils.rgb2lab import _lms2lab, _rgb2lms
import cv2 as cv


_lms2rgb = np.linalg.inv(_rgb2lms)
_lab2lms = np.linalg.inv(_lms2lab)


def lab2rgb(lab):
    return cv.cvtColor(np.clip(lab, 0, 255).astype("uint8"), cv.COLOR_LAB2RGB)


"""
Implementation adapted from https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_conversion/lab_to_rgb.py
"""
def lab2rgbX(lab):
    m, n, c = lab.shape
    
    # get LMS from LAB
    lab = np.reshape(lab, (m * n, c))
    lms = np.dot(_lab2lms, lab.T)

    # get RGB from LMS and reshape to three channel image
    lms = np.exp(lms)
    lms[lms == np.spacing(1)] = 0
    rgb = np.dot(_lms2rgb, lms)
    return np.reshape(rgb.T, (m, n, c))
