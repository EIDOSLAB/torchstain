import os
import cv2
import torchstain
import tensorflow as tf
import time
from skimage.metrics import structural_similarity as ssim
import numpy as np

def test_normalize_tf():
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
    result_numpy = result_numpy.astype("float32")
    result_tf = result_tf.numpy().astype("float32")

    # assess whether the normalized images are identical across backends
    np.testing.assert_almost_equal(ssim(result_numpy.flatten(), result_tf.flatten()), 1.0, decimal=4, verbose=True)
