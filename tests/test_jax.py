import os
import cv2
import torchstain
import torchstain.jax
import time
from skimage.metrics import structural_similarity as ssim
import numpy as np
from jax import numpy as jnp

def test_macenko_jax():
    size = 1024
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    target = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/target.png")), cv2.COLOR_BGR2RGB), (size, size))
    to_transform = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/source.png")), cv2.COLOR_BGR2RGB), (size, size))

    # initialize normalizers for each backend and fit to target image
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target)

    jax_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='jax')
    jax_normalizer.fit(target)

    # transform
    result_numpy, _, _ = normalizer.normalize(I=to_transform)
    result_jax, _, _ = jax_normalizer.normalize(I=to_transform)

    # convert to numpy and set dtype
    result_numpy = result_numpy.astype("float32")
    result_jax = np.asarray(result_jax).astype("float32")

    # assess whether the normalized images are identical across backends
    np.testing.assert_almost_equal(ssim(result_numpy.flatten(), result_jax.flatten()), 1.0, decimal=4, verbose=True)
