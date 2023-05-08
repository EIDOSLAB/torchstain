import tensorflow as tf
from torchstain.base.augmentors.he_augmentor import HEAugmentor
from torchstain.tf.utils import cov, percentile, solveLS
import numpy as np
import tensorflow.keras.backend as K


"""
Source code ported from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class TensorFlowMacenkoAugmentor(HEAugmentor):
    def __init__(self, sigma1=0.2, sigma2=0.2):
        super().__init__()

        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.I = None

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
        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        # keep these as we will use them for augmentation
        self.I = I
        self.HERef = HE
        self.CRef = C
        self.maxCRef = maxC

    def augment(self, Io=240, alpha=1, beta=0.15):
        I = self.I
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = maxC / self.maxCRef
        C2 = C / tf.expand_dims(maxC, axis=-1)

        # introduce noise to the concentrations (applied along axis=0)
        C2 = (C2 * tf.random.uniform((2, 1), 1 - self.sigma1, 1 + self.sigma1)) + tf.random.uniform((2, 1), -self.sigma2, self.sigma2)

        # recreate the image using reference mixing matrix
        Iaug = Io * tf.math.exp(-tf.linalg.matmul(self.HERef, C2))
        Iaug = tf.clip_by_value(Iaug, 0, 255)
        Iaug = tf.cast(tf.reshape(tf.transpose(Iaug), shape=(h, w, c)), tf.int32)

        return Iaug
