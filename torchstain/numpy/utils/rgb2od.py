import numpy as np

# https://github.com/Peter554/StainTools/blob/2089900d11173ee5ea7de95d34532932afd3181a/staintools/utils/optical_density_conversion.py#L4
def rgb2od(I):
    # remove zeros
    I[I == 0] = 1
    
    # convert to OD and return
    return np.maximum(-1 * np.log(I / 255), 1e-6)
