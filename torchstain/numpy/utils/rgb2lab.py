import numpy as np
import cv2 as cv
import skimage


# constant conversion matrices between color spaces
# https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L392
_rgb2xyz = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])

_rgb2lms = _rgb2xyz

# "D65": {'2': (0.95047, 1., 1.08883),   # This was: `lab_ref_white`
#              '10': (0.94809667673716, 1, 1.0730513595166162),
#              'R': (0.9532057125493769, 1, 1.0853843816469158)},

_lms2lab = np.dot(
    np.array([[1 / (3 ** 0.5), 0, 0],
              [0, 1 / (6 ** 0.5), 0],
              [0, 0, 1 / (2 ** 0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]])
)


def rgb2labCV(rgb):
    return cv.cvtColor(rgb.astype("uint8"), cv.COLOR_RGB2LAB)


def rgb2labY(rgb):
    rgb = rgb.astype("float32") / 255
    x = skimage.color.rgb2lab(rgb)

    # OpenCV format
    x[..., 0] *= 2.55
    x[..., 1] += 128
    x[..., 2] += 128
    return x


"""
Implementation adapted from https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""
def rgb2lab(rgb):
    # first normalize
    rgb = rgb.astype("float32") / 255

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


"""
Implementation adapted from https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_conversion/rgb_to_lab.py
"""
def rgb2lab_old(rgb):
    # rgb = rgb.astype("float32") / 255
    rgb = rgb.astype("float32")
    m, n, c = rgb.shape

    # get LMS from RGB
    rgb = np.reshape(rgb, (m * n, c))
    lms = np.dot(_rgb2lms, rgb.T)
    lms[lms == 0] = np.spacing(1)

    # get LAB from LMS and reshape to 3-channel image
    lab = np.dot(_lms2lab, np.log(lms))
    lab = np.reshape(lab.T, (m, n, c))

    return lab
