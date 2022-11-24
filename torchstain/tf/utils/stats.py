import tensorflow as tf

def get_mean_std(I):
    return tf.math.reduce_mean(I), tf.math.reduce_std(I)

def standardize(x, mu, std):
    return (x - mu) / std
