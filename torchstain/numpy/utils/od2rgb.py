import numpy as np

def OD_to_RGB(OD):
    return (255 * np.exp(-1 * OD)).astype(np.uint8)
