import numpy as np

class myPCA:
    def __init__(self, n_components, method = "eigen"):
        
        self.n_components = n_components
        self.method = method
        self.components = None
        self.mean = None
        self.var_explained = None
        
    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)
        
    def transform(self, X):
        X = X - self.mean
        
        if self.method == "svd":
            U, s, Vt = np.linalg.svd(X)
            if self.n_components is not None:
                U = U[:, :self.n_components]
                s = s[:self.n_components]
                Vt = Vt[:self.n_components, :]
            self.components = Vt[:self.n_components].T
            self.var_explained = s / np.sum(s)
          
        elif self.method == "eigen":
            # Calculate the covariance matrix of X
            cov_X = np.cov(X, rowvar=False)
            # Calculate the eigenvalues and eigenvectors of the covariance matrix
            eigvals, eigvecs = np.linalg.eigh(cov_X)
            # Sort the eigenvectors by descending eigenvalues
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # the rate of variance of the components 
            self.var_explained = eigvals / np.sum(eigvals)
            
            # store the first n_components eigenvectors as the principal components
            self.components = eigvecs[:, : self.n_components]
    
    def fit_transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
    
    def inverse_transform(self,X,X_transformed): 
        
        X_reconstructed = X_transformed.dot(self.components.T) + np.mean(X, axis=0)

        
        return X_reconstructed
    
    def explained_variance_ratio(self):
        return self.var_explained
 