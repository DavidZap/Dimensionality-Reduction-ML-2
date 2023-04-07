import numpy as np

class PCA:
    def __init__(self, n_components, method = "svd"):
        
        self.n_components = n_components
        self.method = method
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self._transform(X)
        
    def transform(self, X):
        X = X.copy()
        X = X - self.mean
        
        if self.method == "svd":
            U, s, Vt = np.linalg.svd(X)
            if self.n_components is not None:
                U = U[:, :self.n_components]
                s = S[:self.n_components]
                Vt = V[:self.n_components, :]
            self.components = Vt[:self.n_components].T
            
        else self.method == "eigen":
            # Calculate the covariance matrix of X
            cov_X = np.cov(X.T)
            # Calculate the eigenvalues and eigenvectors of the covariance matrix
            eigvals, eigvecs = np.linalg.eig(cov_X)
            # Sort the eigenvectors by descending eigenvalues
            idx = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:, idx]
            eigvals = eigvals[idx]
            
            # store the first n_components eigenvectors as the principal components
            self.components = eigvecs[:, : self.n_components]
            
    def fit_transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
            
            
            
            
            
            
    
    


def my_svd(X, k=None):
    U, S, V = np.linalg.svd(X)
    if k is not None:
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
    return U, S, V

########### PCA ##################

import numpy as np

def my_pca(X, k=None):
    # Calculate the covariance matrix of X
    cov_X = np.cov(X.T)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov_X)
    # Sort the eigenvectors by descending eigenvalues
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    # Select the first k eigenvectors, if specified
    if k is not None:
        eigvecs = eigvecs[:, :k]
    # Transform the data using the selected eigenvectors
    X_pca = np.dot(X, eigvecs)
    return X_pca
