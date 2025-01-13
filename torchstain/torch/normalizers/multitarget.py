import torch
from torchstain.torch.utils import cov, percentile

"""
Implementation of the multi-target normalizer from the paper: https://arxiv.org/pdf/2406.02077
"""
class TorchMultiMacenkoNormalizer:
    def __init__(self, norm_mode="avg-post"):
        self.norm_mode = norm_mode
        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])
        self.updated_lstsq = hasattr(torch.linalg, "lstsq")
        
    def __convert_rgb2od(self, I, Io, beta):
        I = I.permute(1, 2, 0)
        OD = -torch.log((I.reshape((-1, I.shape[-1])).float() + 1)/Io)
        ODhat = OD[~torch.any(OD < beta, dim=1)]
        return OD, ODhat

    def __find_phi_bounds(self, ODhat, eigvecs, alpha):
        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])

        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)

        return minPhi, maxPhi

    def __find_HE_from_bounds(self, eigvecs, minPhi, maxPhi):
        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        return HE

    def __find_HE(self, ODhat, eigvecs, alpha):
        minPhi, maxPhi = self.__find_phi_bounds(ODhat, eigvecs, alpha)
        return self.__find_HE_from_bounds(eigvecs, minPhi, maxPhi)

    def __find_concentration(self, OD, HE):
        Y = OD.T
        if not self.updated_lstsq:
            return torch.lstsq(Y, HE)[0][:2]
        return torch.linalg.lstsq(HE, Y)[0]

    def __compute_matrices_single(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # _, eigvecs = torch.symeig(cov(ODhat.T), eigenvectors=True)
        _, eigvecs = torch.linalg.eigh(cov(ODhat.T), UPLO='U')
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, Is, Io=240, alpha=1, beta=0.15):
        if self.norm_mode == "avg-post":
            HEs, _, maxCs = zip(*(
                self.__compute_matrices_single(I, Io, alpha, beta)
                for I in Is
            ))

            self.HERef = torch.stack(HEs).mean(dim=0)
            self.maxCRef = torch.stack(maxCs).mean(dim=0)
        elif self.norm_mode == "concat":
            ODs, ODhats = zip(*(
                self.__convert_rgb2od(I, Io, beta)
                for I in Is
            ))
            OD = torch.cat(ODs, dim=0)
            ODhat = torch.cat(ODhats, dim=0)

            eigvecs =  torch.symeig(cov(ODhat.T), eigenvectors=True)[1][:, [1, 2]]

            HE =  self.__find_HE(ODhat, eigvecs, alpha)

            C = self.__find_concentration(OD, HE)
            maxCs = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode == "avg-pre":
            ODs, ODhats = zip(*(
                self.__convert_rgb2od(I, Io, beta)
                for I in Is
            ))
            
            eigvecs = torch.stack([torch.symeig(cov(ODhat.T), eigenvectors=True)[1][:, [1, 2]] for ODhat in ODhats]).mean(dim=0)

            OD = torch.cat(ODs, dim=0)
            ODhat = torch.cat(ODhats, dim=0)
            
            HE =  self.__find_HE(ODhat, eigvecs, alpha)

            C = self.__find_concentration(OD, HE)
            maxCs = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode == "fixed-single" or self.norm_mode == "stochastic-single":
            # single img
            self.HERef, _, self.maxCRef = self.__compute_matrices_single(Is[0], Io, alpha, beta)
        else:
            raise "Unknown norm mode"

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices_single(I, Io, alpha, beta)
        C = (self.maxCRef / maxC).unsqueeze(-1) * C

        Inorm = Io * torch.exp(-torch.matmul(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.T.reshape(h, w, c).int()

        H, E = None, None

        if stains:
            H = torch.mul(Io, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()

            E = torch.mul(Io, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()

        return Inorm, H, E
