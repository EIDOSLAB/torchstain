from torchstain.numpy.utils.rgb2lab import rgb2lab
from torchstain.numpy.utils.lab2rgb import lab2rgb
import numpy as np
import cv2
import os

def test_rgb_lab():
    size = 1024
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/source.png")), cv2.COLOR_BGR2RGB), (size, size))
    
    # rgb2lab expects data to be float32 in range [0, 1]
    img = img / 255

    # convert from RGB to LAB and back again to RGB
    reconstructed_img = lab2rgb(rgb2lab(img))

    # assess if the reconstructed image is similar to the original image
    np.testing.assert_almost_equal(np.mean(np.abs(reconstructed_img - img)), 0.0, decimal=4, verbose=True)
