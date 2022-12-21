import numpy as np


def RGB_to_OD(I):
    # remove zeros
    x[x == 0] = 1
    
    # convert to OD and return
    return -1 * np.log(I / 255)