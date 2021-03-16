import numpy as np
import scipy.spatial 

def median_trick(X, scale = 1.):
    '''
    X shap: N x d, where N is the number of samples, and d is the dimension of sample
    return: gamma for exp(- gamma \|x - y\|^2)
    '''
    N = X.shape[0]
    pdists = scipy.spatial.distance.pdist(X)
    return 1./((scale*np.median(pdists))**2)

    


