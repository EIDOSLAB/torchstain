import tensorflow as tf
from torchstain.base.normalizers.he_normalizer import HENormalizer
from torchstain.tf.utils import cov, percentile, solveLS
import numpy as np
import tensorflow.keras.backend as K


"""
Source code ported from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class TensorFlowMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = tf.constant([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = tf.constant([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io, beta):
        I = tf.transpose(I, [1, 2, 0])

        # calculate optical density
        OD = -tf.math.log((tf.cast(tf.reshape(I, [tf.math.reduce_prod(I.shape[:-1]), I.shape[-1]]), tf.float32) + 1) / Io)

        # remove transparent pixels
        ODhat = OD[~tf.math.reduce_any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = tf.linalg.matmul(ODhat, eigvecs)
        phi = tf.math.atan2(That[:, 1], That[:, 0])

        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)

        vMin = tf.matmul(eigvecs, tf.expand_dims(tf.stack((tf.math.cos(minPhi), tf.math.sin(minPhi))), axis=-1))
        vMax = tf.matmul(eigvecs, tf.expand_dims(tf.stack((tf.math.cos(maxPhi), tf.math.sin(maxPhi))), axis=-1))

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        HE = tf.where(vMin[0] > vMax[0], tf.concat((vMin, vMax), axis=1), tf.concat((vMax, vMin), axis=1))

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = tf.transpose(OD)

        # determine concentrations of the individual stains
        return solveLS(HE, Y)[:2]

    def __compute_matrices(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = tf.linalg.eigh(cov(tf.transpose(ODhat)))
        eigvecs = eigvecs[:, 1:3]

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        maxC = tf.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        ''' Normalize staining appearence of H&E stained images

        Example use:
            see test.py

        Input:
            I: RGB input image: tensor of shape [C, H, W] and type uint8
            Io: (optional) transmitted light intensity
            alpha: percentile
            beta: transparency threshold
            stains: if true, return also H & E components

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        # normalize stain concentrations
        C *= tf.expand_dims((self.maxCRef / maxC), axis=-1)

        # recreate the image using reference mixing matrix
        Inorm = Io * tf.math.exp(-tf.linalg.matmul(self.HERef, C))
        Inorm = tf.clip_by_value(Inorm, 0, 255)
        Inorm = tf.cast(tf.reshape(tf.transpose(Inorm), shape=(h, w, c)), tf.int32)

        H, E = None, None

        if stains:
            H = tf.math.multiply(Io, tf.math.exp(tf.linalg.matmul(-tf.expand_dims(self.HERef[:, 0], axis=-1), tf.expand_dims(C[0, :], axis=0))))
            H = tf.clip_by_value(H, 0, 255)
            H = tf.cast(tf.reshape(tf.transpose(H), shape=(h, w, c)), tf.int32)

            E = tf.math.multiply(Io, tf.math.exp(tf.linalg.matmul(-tf.expand_dims(self.HERef[:, 1], axis=-1), tf.expand_dims(C[1, :], axis=0))))
            E = tf.clip_by_value(E, 0, 255)
            E = tf.cast(tf.reshape(tf.transpose(E), shape=(h, w, c)), tf.int32)

        return Inorm, H, E
