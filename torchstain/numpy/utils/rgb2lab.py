import numpy as np

# constant conversion matrices between color spaces: https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
_rgb2xyz = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])

"""
Implementation adapted from:
https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""
def rgb2lab(rgb):
    rgb = rgb.astype("float32")

    # convert rgb -> xyz color domain
    arr = rgb.copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    xyz = np.dot(arr, _rgb2xyz.T.astype(arr.dtype))

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = xyz.copy()
    arr = arr / np.asarray((0.95047, 1., 1.08883), dtype=xyz.dtype)

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = np.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

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
