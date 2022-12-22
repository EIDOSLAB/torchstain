import numpy as np
from torchstain.numpy.utils.rgb2od import rgb2od
from torchstain.numpy.utils.rgb2lab import rgb2lab
from torchstain.numpy.utils.lasso import lasso
import spams
from sklearn.linear_model import LassoLars
from sklearn.decomposition import DictionaryLearning

def extract_tissue(I, th=0.8):
    LAB = rgb2lab(I / 255)
    L = LAB[:, :, 0] / 255.0
    return L < th

def get_stain_matrix(I, th=0.8, alpha=0.1):
    # convert RGB -> OD and flatten channel-wise
    OD = rgb2od(I).reshape((-1, 3))

    # detect glass and remove it from OD image
    mask = extract_tissue(I, th).reshape((-1,))
    OD = OD[mask]

    # perform dictionary learning @TODO: Implement DL train method
    #model = DictionaryLearning(
    #    fit_algorithm="lars",
    #    transform_algorithm="lasso_lars", n_components=2, transform_n_nonzero_coefs=0,
    #    transform_alpha=alpha, positive_dict=True, verbose=False, split_sign=True,
    #    # positive_code=True,
    #)
    #dictionary1 = model.fit_transform(OD)
    #print(dictionary1)
    dictionary = spams.trainDL(OD.T, K=2, lambda1=alpha, mode=2, modeD=0,
                                posAlpha=True, posD=True, verbose=False)
    
    #print(dictionary)
    #exit()
    dictionary = dictionary.T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    
    # normalize rows and return result
    return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

def get_concentrations(I, stain_matrix, alpha=0.01):
    # convert RGB -> OD and flatten channel-wise
    OD = rgb2od(I).reshape((-1, 3))

    # perform LASSO regression
    #model = LassoLars(alpha=alpha, positive=True, fit_intercept=False)
    #model.fit(X=OD.T, y=stain_matrix.T)
    #print(OD.T)
    #return model.predict(OD.T).T
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=alpha, pos=True).toarray().T
    #return lasso(OD.T, y=stain_matrix.T).T  # @TODO: Implement LARS-LASSO
