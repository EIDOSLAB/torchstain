import tensorflow as tf
from torchstain.base.normalizers import HENormalizer
from torchstain.tf.utils.rgb2lab import rgb2lab
from torchstain.tf.utils.lab2rgb import lab2rgb
from torchstain.tf.utils.split import csplit, cmerge, lab_split, lab_merge
from torchstain.tf.utils.stats import get_mean_std, standardize

"""
Source code adapted from:
https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_normalization/reinhard.py
https://github.com/Peter554/StainTools/blob/master/staintools/reinhard_color_normalizer.py
"""
class TensorFlowReinhardNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()
        self.target_mus = None
        self.target_stds = None
    
    def fit(self, target):
        # normalize
        target = tf.cast(target, tf.float32) / 255

        # convert to LAB
        lab = rgb2lab(target)

        # get summary statistics
        stack_ = tf.convert_to_tensor([get_mean_std(x) for x in lab_split(lab)])
        self.target_means = stack_[:, 0]
        self.target_stds = stack_[:, 1]

    def normalize(self, I):
        # normalize
        I = tf.cast(I, tf.float32) / 255
        
        # convert to LAB
        lab = rgb2lab(I)
        labs = lab_split(lab)

        # get summary statistics from LAB
        stack_ = tf.convert_to_tensor([get_mean_std(x) for x in labs])
        mus = stack_[:, 0]
        stds = stack_[:, 1]

        # standardize intensities channel-wise and normalize using target mus and stds
        result = [standardize(x, mu_, std_) * std_T + mu_T for x, mu_, std_, mu_T, std_T \
            in zip(labs, mus, stds, self.target_means, self.target_stds)]
        
        # rebuild LAB
        lab = lab_merge(*result)

        # convert back to RGB from LAB
        lab = lab2rgb(lab)

        # rescale to [0, 255] uint8
        return tf.cast(lab * 255, tf.uint8)
