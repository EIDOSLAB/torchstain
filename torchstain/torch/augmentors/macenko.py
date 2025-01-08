import torch
from torchstain.base.augmentors.he_augmentor import HEAugmentor
from torchstain.torch.utils import cov, percentile

"""
Source code ported from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class TorchMacenkoAugmentor(HEAugmentor):
    def __init__(self, sigma1=0.2, sigma2=0.2):
        super().__init__()

        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.I = None

        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

        # Avoid using deprecated torch.lstsq (since 1.9.0)
        self.updated_lstsq = hasattr(torch.linalg, 'lstsq')
        
    def __convert_rgb2od(self, I, Io, beta):
        I = I.permute(1, 2, 0)

        # calculate optical density
        OD = -torch.log((I.reshape((-1, I.shape[-1])).float() + 1)/Io)

        # remove transparent pixels
        ODhat = OD[~torch.any(OD < beta, dim=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])

        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)

        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = OD.T

        # determine concentrations of the individual stains
        if not self.updated_lstsq:
            return torch.lstsq(Y, HE)[0][:2]
    
        return torch.linalg.lstsq(HE, Y)[0]

    def __compute_matrices(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = torch.linalg.eigh(cov(ODhat.T)) 
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        # keep these as we will use them for augmentation
        self.I = I
        self.HERef = HE
        self.CRef = C
        self.maxCRef = maxC
    
    @staticmethod
    def random_uniform(shape, low, high):
        return (low - high) * torch.rand(*shape) + high

    def augment(self, Io=240, alpha=1, beta=0.15):
        I = self.I
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = maxC / self.maxCRef
        C2 = C / torch.unsqueeze(maxC, axis=-1)

        # introduce noise to the concentrations (applied along axis=0)
        C2 = (C2 * self.random_uniform((2, 1), 1 - self.sigma1, 1 + self.sigma1)) + self.random_uniform((2, 1), -self.sigma2, self.sigma2)

        # recreate the image using reference mixing matrix
        Inorm = Io * torch.exp(-torch.matmul(self.HERef, C2))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.T.reshape(h, w, c).int()

        return Inorm
        