from typing import Union
import tensorflow as tf

def percentile(t: tf.Tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    k = 1 + tf.math.round(.01 * tf.cast(q, tf.float32) * (tf.cast(tf.math.reduce_prod(tf.size(t)), tf.float32) - 1))
    return tf.sort(tf.reshape(t, [-1]))[tf.cast(k - 1, tf.int32)]
