import torch

def cov(x):
    """
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    E_x = x.mean(dim=1)
    x = x - E_x[:, None]
    return torch.mm(x, x.T) / (x.size(1) - 1)
