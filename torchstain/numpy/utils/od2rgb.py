import numpy as np

# https://github.com/Peter554/StainTools/blob/2089900d11173ee5ea7de95d34532932afd3181a/staintools/utils/optical_density_conversion.py#L18
def od2rgb(OD):
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)
