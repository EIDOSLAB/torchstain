import numpy as np

def get_mean_std(I):
    return np.mean(I), np.std(I)

def standardize(x, mu, std):
    return (x - mu) / std

def standardize_brightness(x, alpha=99):
    p = np.percentile(x, alpha)
    return np.clip(x * 255 / p, 0, 255).astype("uint8")
