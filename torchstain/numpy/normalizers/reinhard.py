import numpy as np
from torchstain.base.normalizers import HENormalizer
from torchstain.numpy.utils.rgb2lab import rgb2lab
from torchstain.numpy.utils.stats import get_mean_std, standardize

"""
Source code adapted from:
https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_normalization/reinhard.py
https://github.com/Peter554/StainTools/blob/master/staintools/reinhard_color_normalizer.py
"""
class NumpyReinhardNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()
        # unless fit() is applied, no transfer is performed
        self.target_mus = np.ones(3)
        self.target_stds = np.ones(3)
    
    def fit(self, target):
        self.target_mus, self.target_stds = get_mean_std(target)

    def normalize(self, I):
        # convert to LAB
        lab = rgb2lab(I)

        # get summary statistics from LAB
        mus, stds = get_mean_std(lab)

        # standardize intensities channel-wise and normalize using target means and standard deviations (one for each channel)
        return np.dstack([standardize(x, mu_, std_) for x, mu, std_, mu_T, std_T in zip(csplit(lab), mus, stds, target_mus, target_stds)])
