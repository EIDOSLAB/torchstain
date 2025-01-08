import numpy as np
from torchstain.numpy.utils.rgb2lab import _rgb2xyz

_xyz2rgb = np.linalg.inv(_rgb2xyz)

"""
Implementation is based on:
https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""
def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert an array of LAB values to RGB values.

    Args:
        lab (np.ndarray): An array of shape (..., 3) containing LAB values.

    Returns:
        np.ndarray: An array of shape (..., 3) containing RGB values.
    """
    # first rescale back from OpenCV format
    lab[..., 0] /= 2.55
    lab[..., 1:] -= 128

    # convert LAB -> XYZ color domain
    y = (lab[..., 0] + 16.) / 116.
    x = (lab[..., 1] / 500.) + y
    z = y - (lab[..., 2] / 200.)

    xyz = np.stack([x, y, z], axis=-1)

    mask = xyz > 0.2068966
    xyz[mask] = np.power(xyz[mask], 3.)
    xyz[~mask] = (xyz[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz *= np.array((0.95047, 1., 1.08883), dtype=xyz.dtype)

    # convert XYZ -> RGB color domain
    rgb = np.matmul(xyz, _xyz2rgb.T)

    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
    rgb[~mask] *= 12.92

    return np.clip(rgb, 0, 1)
