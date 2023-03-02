import os
import cv2
import torchstain
import torchstain.tf
import tensorflow as tf
import numpy as np

def test_cov():
    x = np.random.randn(10, 10)
    cov_np = np.cov(x)
    cov_t = torchstain.tf.utils.cov(x)

    np.testing.assert_almost_equal(cov_np, cov_t.numpy())

def test_percentile():
    x = np.random.randn(10, 10)
    p = 20
    p_np = np.percentile(x, p, interpolation='nearest')
    p_t = torchstain.tf.utils.percentile(x, p)

    np.testing.assert_almost_equal(p_np, p_t)

def test_macenko_tf():
    size = 1024
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    target = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/target.png")), cv2.COLOR_BGR2RGB), (size, size))
    to_transform = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/source.png")), cv2.COLOR_BGR2RGB), (size, size))

    # setup preprocessing and preprocess image to be normalized
    T = lambda x: tf.convert_to_tensor(np.moveaxis(x, -1, 0).astype("float32"))  #  * 255
    t_to_transform = T(to_transform)

    # initialize normalizers for each backend and fit to target image
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target)

    tf_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='tensorflow')
    tf_normalizer.fit(T(target))

    # transform
    result_numpy, _, _ = normalizer.normalize(I=to_transform, stains=True)
    result_tf, _, _ = tf_normalizer.normalize(I=t_to_transform, stains=True)

    # convert to numpy and set dtype
    result_numpy = result_numpy.astype("float32") / 255.
    result_tf = result_tf.numpy().astype("float32") / 255.

    # assess whether the normalized images are identical across backends
    np.testing.assert_almost_equal(result_numpy.flatten(), result_tf.flatten(), decimal=2, verbose=True)

def test_reinhard_tf():
    size = 1024
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    target = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/target.png")), cv2.COLOR_BGR2RGB), (size, size))
    to_transform = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/source.png")), cv2.COLOR_BGR2RGB), (size, size))

    # setup preprocessing and preprocess image to be normalized
    T = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
    t_to_transform = T(to_transform)

    # initialize normalizers for each backend and fit to target image
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='numpy')
    normalizer.fit(target)

    tf_normalizer = torchstain.normalizers.ReinhardNormalizer(backend='tensorflow')
    tf_normalizer.fit(T(target))

    # transform
    result_numpy = normalizer.normalize(I=to_transform)
    result_tf = tf_normalizer.normalize(I=t_to_transform)

    # convert to numpy and set dtype
    result_numpy = result_numpy.astype("float32") / 255.
    result_tf = result_tf.numpy().astype("float32") / 255.

    # assess whether the normalized images are identical across backends
    np.testing.assert_almost_equal(result_numpy.flatten(), result_tf.flatten(), decimal=2, verbose=True)
