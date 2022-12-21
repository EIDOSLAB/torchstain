import numpy as np
from torchstain.base.normalizers import HENormalizer
from torchstain.numpy.utils.lars import lars
from torchstain.numpy.utils.stats import standardize_brightness
from torchstain.numpy.utils.extract import get_stain_matrix

"""
Source code adapted from: https://github.com/wanghao14/Stain_Normalization/blob/master/stainNorm_Vahadane.py
"""
class NumpyVahadaneNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()
        self.stain_matrix_target = None
    
    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def normalize(self, I):
        I = standardize_brightness(I)
        stain_matrix = get_stain_matrix(I)
        concentrations = get_concentrations(I, stain_matrix)
        out = np.exp(-1 * np.dot(concentrations, self.stain_matrix_target).reshape(I.shape))
        return (255 * out).astype("uint8")



