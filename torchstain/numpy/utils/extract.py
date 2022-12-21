import numpy as np
from torchstain.numpy.utils.rgb2od import rgb2od
from torchstain.numpy.utils.rgb2lab import rgb2lab
from torchstain.numpy.utils.lasso import lasso

def extract_tissue(I, th=0.8):
    LAB = rgb2lab(I)
    L = I_LAB[:, :, 0] / 255.0
    return L < th

def get_stain_matrix(I, th=0.8, lamda=0.1):
    # convert RGB -> OD and flatten channel-wise
    OD = rgb2od(I).reshape((-1, 3))

    # detect glass and remove it from OD image
    mask = extract_tissue(I, th).reshape((-1,))
    OD = OD[mask]

    # perform dictionary learning
    # dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False)
    dictionary = None  # @TODO: Implement DL train method
    dictionary = dictionary.T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    
    # normalize rows and return result
    return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

def get_concentrations(I, stain_matrix, lamda=0.01):
    # convert RGB -> OD and flatten channel-wise
    OD = rgb2od(I).reshape((-1, 3))

    # perform LASSO regression
    #return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T
    return lasso(OD.T, y=stain_matrix.T).T  # @TODO: Implement LARS-LASSO
