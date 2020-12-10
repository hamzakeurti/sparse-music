import numpy as np

def whiten_fit(X,threshold = None, n_components = None):
    # X expected in the shape (m features x n samples)
    assert(X.shape[0] < X.shape[1])
    # Center
    mean_vec = np.mean(X,axis=1)[:,None]
    X_centered = X - mean_vec
    
    # Covariance matrix
    C = np.corrcoef(X_centered,rowvar = True)    
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    eigen_vals = np.real(eigen_vals)
    eigen_vecs = np.real(eigen_vecs)
    if np.sum(eigen_vals<=0) >0:
        epsilon = 1e-6
        C = C + epsilon*np.identity(C.shape[0])
    
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    eigen_vals = np.real(eigen_vals)
    eigen_vecs = np.real(eigen_vecs)
    
    
    
    # Choose number of components
    if not n_components:
        if not threshold:
            n_components = X.shape[0]
        else:
            normalized_cumsum = np.cumsum(eigen_vals)/np.sum(eigen_vals)
            for i in range(X.shape[0]):
                if normalized_cumsum[i] > threshold:
                    n_components = i
                    break
    
    return eigen_vecs,eigen_vals,mean_vec

def whiten_fit_transform(X,threshold = None, n_components = None):
    eigen_vecs,eigen_vals,mean_vec = whiten_fit(X,threshold, n_components)
    X_centered = X - mean_vec

    # Project X
    X_pca = np.dot(eigen_vecs[:,:(n_components+1)].T,X_centered)
    
    # Whiten X
    X_whitened = np.dot(np.diag(1/(eigen_vals[:(n_components+1)]**.5)),X_pca)
    
    
    return X_whitened,eigen_vecs,eigen_vals,mean_vec


def whiten_transform(X,eigen_vecs,eigen_vals,mean_vec,n_components = None):
    # Choose number of components
    if not n_components:
        n_components = X.shape[0]
        
    X_centered = X - mean_vec
    # Project X
    X_pca = np.dot(eigen_vecs[:,:(n_components)].T,X_centered)
    
    # Whiten X
    X_whitened = np.dot(np.diag(1/(eigen_vals[:(n_components)]**.5)),X_pca)
    
    return X_whitened

def reconstruct_from_whitened(X_whitened,eigen_vecs,eigen_vals,mean_vec):
    D2 = np.diag(eigen_vals[:X_whitened.shape[0]]**.5)
    X = np.dot(D2,X_whitened)
    
    X = np.dot(eigen_vecs[:,:X_whitened.shape[0]],X)
    X = X+mean_vec
    return X

def keep_n_components(eigen_vals,threshold):

    normalized_cumsum = np.cumsum(eigen_vals)/np.sum(eigen_vals)
    for i in range(eigen_vals.shape[0]):
        if normalized_cumsum[i] > threshold:
            n_components = i
            break
    
    return n_components