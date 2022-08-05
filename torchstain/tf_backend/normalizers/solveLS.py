import tensorflow as tf

def solveLS(A, B):
    q_full, _ = tf.linalg.qr(A, full_matrices=True)
    ret1 = tf.linalg.lstsq(A, B)
    ret2 = tf.linalg.lstsq(q_full, B)
    return tf.concat([ret1, ret2[ret1.shape[0]:]], axis=0)
