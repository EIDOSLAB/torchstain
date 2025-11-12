import numpy as np

class NumpyMultiMacenkoNormalizer:
    def __init__(self, norm_mode="avg-post"):
        self.norm_mode = norm_mode
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io, beta):
        I = np.transpose(I, (1, 2, 0))
        OD = -np.log((I.reshape(-1, I.shape[-1]).astype(float) + 1) / Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]
        return OD, ODhat

    def __find_phi_bounds(self, ODhat, eigvecs, alpha):
        That = np.dot(ODhat, eigvecs)
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        return minPhi, maxPhi

    def __find_HE_from_bounds(self, eigvecs, minPhi, maxPhi):
        vMin = np.dot(eigvecs, [np.cos(minPhi), np.sin(minPhi)]).reshape(-1, 1)
        vMax = np.dot(eigvecs, [np.cos(maxPhi), np.sin(maxPhi)]).reshape(-1, 1)

        HE = np.concatenate([vMin, vMax], axis=1) if vMin[0] > vMax[0] else np.concatenate([vMax, vMin], axis=1)
        return HE

    def __find_HE(self, ODhat, eigvecs, alpha):
        minPhi, maxPhi = self.__find_phi_bounds(ODhat, eigvecs, alpha)
        return self.__find_HE_from_bounds(eigvecs, minPhi, maxPhi)

    def __find_concentration(self, OD, HE):
        Y = OD.T
        C, _, _, _ = np.linalg.lstsq(HE, Y, rcond=None)
        return C

    def __compute_matrices_single(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io, beta)

        cov_matrix = np.cov(ODhat.T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)
        C = self.__find_concentration(OD, HE)
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, Is, Io=240, alpha=1, beta=0.15):
        if self.norm_mode == "avg-post":
            HEs, _, maxCs = zip(*[self.__compute_matrices_single(I, Io, alpha, beta) for I in Is])

            self.HERef = np.mean(HEs, axis=0)
            self.maxCRef = np.mean(maxCs, axis=0)
        elif self.norm_mode == "concat":
            ODs, ODhats = zip(*[self.__convert_rgb2od(I, Io, beta) for I in Is])
            OD = np.vstack(ODs)
            ODhat = np.vstack(ODhats)

            cov_matrix = np.cov(ODhat.T)
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvecs = eigvecs[:, [1, 2]]

            HE = self.__find_HE(ODhat, eigvecs, alpha)
            C = self.__find_concentration(OD, HE)
            maxCs = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode == "avg-pre":
            ODs, ODhats = zip(*[self.__convert_rgb2od(I, Io, beta) for I in Is])

            covs = [np.cov(ODhat.T) for ODhat in ODhats]
            eigvecs = np.mean([np.linalg.eigh(cov)[1][:, [1, 2]] for cov in covs], axis=0)

            OD = np.vstack(ODs)
            ODhat = np.vstack(ODhats)

            HE = self.__find_HE(ODhat, eigvecs, alpha)
            C = self.__find_concentration(OD, HE)
            maxCs = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode in ["fixed-single", "stochastic-single"]:
            self.HERef, _, self.maxCRef = self.__compute_matrices_single(Is[0], Io, alpha, beta)
        else:
            raise ValueError("Unknown norm mode")

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices_single(I, Io, alpha, beta)
        C = (self.maxCRef / maxC).reshape(-1, 1) * C

        Inorm = Io * np.exp(-np.dot(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = np.transpose(Inorm, (1, 0)).reshape(h, w, c).astype(np.int32)

        H, E = None, None

        if stains:
            H = Io * np.exp(-np.dot(self.HERef[:, 0].reshape(-1, 1), C[0, :].reshape(1, -1)))
            H[H > 255] = 255
            H = np.transpose(H, (1, 0)).reshape(h, w, c).astype(np.int32)

            E = Io * np.exp(-np.dot(self.HERef[:, 1].reshape(-1, 1), C[1, :].reshape(1, -1)))
            E[E > 255] = 255
            E = np.transpose(E, (1, 0)).reshape(h, w, c).astype(np.int32)

        return Inorm, H, E
