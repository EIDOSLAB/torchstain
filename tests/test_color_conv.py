from torchstain.numpy.utils.rgb2lab import rgb2lab
from torchstain.numpy.utils.lab2rgb import lab2rgb
import numpy as np
import cv2
import os

def test_rgb_to_lab():
    size = 1024
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(curr_file_path, "../data/source.png")), cv2.COLOR_BGR2RGB), (size, size))

    reconstructed_img = lab2rgb(rgb2lab(img))
    val = np.mean(np.abs(reconstructed_img - img))
    print("MAE:", val)
    assert val < 0.1
