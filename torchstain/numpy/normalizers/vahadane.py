import numpy as np
from torchstain.base.normalizers import HENormalizer
from torchstain.numpy.utils.lasso import lasso
from torchstain.numpy.utils.stats import standardize_brightness
from torchstain.numpy.utils.extract import get_stain_matrix, get_concentrations
from torchstain.numpy.utils.od2rgb import od2rgb

"""
Source code adapted from:
https://github.com/wanghao14/Stain_Normalization/blob/master/stainNorm_Vahadane.py
https://github.com/Peter554/StainTools/blob/master/staintools/stain_normalizer.py
"""
class NumpyVahadaneNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()
        self.stain_matrix_target = None
        self.maxC_target = None
    
    def fit(self, target):
        # target = target.astype("float32")
        self.stain_matrix_target = get_stain_matrix(target)
        concentration_target = get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(concentration_target, 99, axis=0).reshape((1, 2))

    def normalize(self, I):
        # I = I.astype("float32")
        # I = standardize_brightness(I)
        stain_matrix = get_stain_matrix(I)
        concentrations = get_concentrations(I, stain_matrix)
        maxC = np.percentile(concentrations, 99, axis=0).reshape((1, 2))
        concentrations *= (self.maxC_target / maxC)
        out = 255 * np.exp(-1 * np.dot(concentrations, self.stain_matrix_target))
        return out.reshape(I.shape).astype("uint8")
