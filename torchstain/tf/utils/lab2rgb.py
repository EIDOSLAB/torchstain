import tensorflow as tf
from torchstain.tf.utils.rgb2lab import _rgb2xyz, _white

_xyz2rgb = tf.linalg.inv(_rgb2xyz)

def lab2rgb(lab):
    lab = tf.cast(lab, tf.float32)
    
    # rescale back from OpenCV format and extract LAB channel
    L, a, b = lab[..., 0] / 2.55, lab[..., 1] - 128, lab[..., 2] - 128

    # vector scaling to produce X, Y, Z
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    # merge back to get reconstructed XYZ color image
    out = tf.stack([x, y, z], axis=-1)

    # apply boolean transforms
    mask = out > 0.2068966
    not_mask = tf.math.logical_not(mask)
    out = tf.tensor_scatter_nd_update(out, tf.where(mask), tf.pow(tf.boolean_mask(out, mask), 3))
    out = tf.tensor_scatter_nd_update(out, tf.where(not_mask), (tf.boolean_mask(out, not_mask) - 16 / 116) / 7.787)

    # rescale to the reference white (illuminant)
    out = out * tf.cast(_white, out.dtype)
    
    # convert XYZ -> RGB color domain
    arr = tf.identity(out)
    arr = arr @ tf.transpose(_xyz2rgb)
    mask = arr > 0.0031308
    not_mask = tf.math.logical_not(mask)
    arr = tf.tensor_scatter_nd_update(arr, tf.where(mask), 1.055 * tf.pow(tf.boolean_mask(arr, mask), 1 / 2.4) - 0.055)
    arr = tf.tensor_scatter_nd_update(arr, tf.where(not_mask), tf.boolean_mask(out, not_mask) * 12.92)
    return tf.clip_by_value(arr, 0, 1)
