import numpy as np
import cv2 as cv
import skimage


# constant conversion matrices between color spaces
_rgb2lms = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])

_lms2lab = np.dot(
    np.array([[1 / (3 ** 0.5), 0, 0],
              [0, 1 / (6 ** 0.5), 0],
              [0, 0, 1 / (2 ** 0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]])
)


def rgb2lab(rgb):
    return cv.cvtColor(rgb.astype("uint8"), cv.COLOR_RGB2LAB)


def rgb2labX(rgb):
    return skimage.color.rgb2lab(rgb)


"""
Implementation adapted from https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_conversion/rgb_to_lab.py
"""
def rgb2labY(rgb):
    # rgb = rgb.astype("float32")
    m, n, c = rgb.shape

    # get LMS from RGB
    rgb = np.reshape(rgb, (m * n, c))
    lms = np.dot(_rgb2lms, rgb.T)
    lms[lms == 0] = np.spacing(1)

    # get LAB from LMS and reshape to 3-channel image
    lab = np.dot(_lms2lab, np.log(lms))
    lab = np.reshape(lab.T, (m, n, c))

    return lab[..., 0] * 255/100, lab[..., 1] + 128, lab[..., 2] + 128
