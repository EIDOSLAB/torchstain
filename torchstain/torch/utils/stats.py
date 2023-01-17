import torch

def get_mean_std(I):
    return torch.mean(I), torch.std(I)

def standardize(x, mu, std):
    return (x - mu) / std
