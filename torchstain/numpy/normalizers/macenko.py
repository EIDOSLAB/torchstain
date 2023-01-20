import numpy as np
from torchstain.base.normalizers import HENormalizer

"""
Source code adapted from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class NumpyMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -np.log((I.astype(float)+1)/Io)

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

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

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
        I = I.reshape((-1,3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])

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
            I: RGB input image
            Io: (optional) transmitted light intensity

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        h, w, c = I.shape
        I = I.reshape((-1,3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, maxC[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = np.reshape(Inorm.T, (h, w, c)).astype(np.uint8)


        H, E = None, None

        if stains:
            # unmix hematoxylin and eosin
            H = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
            H[H > 255] = 255
            H = np.reshape(H.T, (h, w, c)).astype(np.uint8)

            E = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
            E[E > 255] = 255
            E = np.reshape(E.T, (h, w, c)).astype(np.uint8)

        return Inorm, H, E
