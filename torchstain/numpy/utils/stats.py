import numpy as np

def get_mean_std(I):
    return np.mean(I), np.std(I)

def standardize(x, mu, std):
    return (x - mu) / std
