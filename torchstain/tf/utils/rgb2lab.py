import tensorflow as tf

# constant conversion matrices between color spaces: https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
_rgb2xyz = tf.constant([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]])
                        
_white = tf.constant([0.95047, 1., 1.08883])

def rgb2lab(rgb):
    arr = tf.cast(rgb, tf.float32)

    # convert rgb -> xyz color domain
    mask = arr > 0.04045
    not_mask = tf.math.logical_not(mask)
    arr = tf.tensor_scatter_nd_update(arr, tf.where(mask), tf.math.pow((tf.boolean_mask(arr, mask) + 0.055) / 1.055, 2.4))
    arr = tf.tensor_scatter_nd_update(arr, tf.where(not_mask), tf.boolean_mask(arr, not_mask) / 12.92)

    xyz = arr @ tf.cast(tf.transpose(_rgb2xyz), arr.dtype)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = tf.identity(xyz)
    arr = arr / tf.cast(_white, xyz.dtype)

    # nonlinear distortion and linear transformation
    mask = arr > 0.008856
    not_mask = tf.math.logical_not(mask)
    arr = tf.tensor_scatter_nd_update(arr, tf.where(mask), tf.math.pow(tf.boolean_mask(arr, mask), 1.0 / 3.0))
    arr = tf.tensor_scatter_nd_update(arr, tf.where(not_mask), 7.787 * tf.boolean_mask(arr, not_mask) + 16 / 116)

    # get each channel as individual tensors
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    # OpenCV format
    L *= 2.55
    a += 128
    b += 128

    # finally, get LAB color domain
    return tf.stack([L, a, b], axis=-1)
