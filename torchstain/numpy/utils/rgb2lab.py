import numpy as np
from colour.models import illuminants

"""
Implementation adapted from:
https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""

_rgb2xyz = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])

def rgb2lab(rgb):
    """
    Convert RGB color space to CIELAB color space.

    :param rgb: numpy array of shape (n, 3) containing RGB values in the range [0, 255]
    :return: numpy array of shape (n, 3) containing LAB values
    """
    rgb = rgb.astype("float32") / 255.0

    # convert rgb -> xyz color domain
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] /= 12.92
    xyz = np.dot(rgb, _rgb2xyz.T.astype(rgb.dtype))

    # scale by CIE XYZ tristimulus values of the reference white point
    xyz = xyz / illuminants.illuminant_D65()[np.newaxis, :]

    # Nonlinear distortion and linear transformation
    mask = xyz > 0.008856
    xyz[mask] = np.cbrt(xyz[mask])
    xyz[~mask] = 7.787 * xyz[~mask] + 16. / 116.

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    # OpenCV format
    L *= 2.55
    a += 128
    b += 128

    # finally, get LAB color domain
    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)
