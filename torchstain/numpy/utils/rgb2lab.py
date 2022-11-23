import numpy as np


# constant conversion matrices between color spaces
_rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])

_lms2lab = np.dot(
    np.array([[1 / (3 ** 0.5), 0, 0],
              [0, 1 / (6 ** 0.5), 0],
              [0, 0, 1 / (2 ** 0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]])
)


"""
Implementation adapted from https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_conversion/rgb_to_lab.py
"""
def rgb2lab(I):
    m, n = I.shape[:2]
    
    # get LMS from RGB
    rgb = np.reshape(I, (m * n, 3))
    lms = np.dot(_rgb2lms, rgb.T)
    lms[lms == 0] = np.spacing(1)

    # get LAB from LMS and reshape to three channel image
    lab = np.dot(_lms2lab, np.log(lms))
    return np.reshape(lab.T, (m, n, 3))
