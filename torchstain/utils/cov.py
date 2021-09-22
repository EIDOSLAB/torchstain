import torch
import tensorflow as tf

def cov(x):
    """
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    E_x = x.mean(dim=1)
    x = x - E_x[:, None]
    return torch.mm(x, x.T) / (x.size(1) - 1)


def cov_tf(x):
    """
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    E_x = tf.math.reduce_mean(x, axis=1)
    x = x - E_x[:, None]
    return tf.linalg.matmul(x, tf.transpose(x)) / (x.shape[1] - 1)
