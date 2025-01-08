import numpy as np
from torchstain.base.augmentors import HEAugmentor

"""
Source code adapted from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class NumpyMacenkoAugmentor(HEAugmentor):
    def __init__(self, sigma1=0.2, sigma2=0.2):
        super().__init__()

        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.I = None

        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -np.log((I.astype(float) + 1) / Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        #project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:,1:3])

        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1, 3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:], 99)])

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
        h, w, c = I.shape
        I = I.reshape((-1, 3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, maxC[:, np.newaxis])

        # introduce noise to the concentrations
        for i in range(C2.shape[0]):
            C2[i, :] *= np.random.uniform(1 - self.sigma1, 1 + self.sigma1)  # multiplicative
            C2[i, :] += np.random.uniform(-self.sigma2, self.sigma2)  # additative

        # recreate the image using reference mixing matrix
        Iaug = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Iaug[Iaug > 255] = 255
        Iaug = np.reshape(Iaug.T, (h, w, c)).astype(np.uint8)

        return Iaug
