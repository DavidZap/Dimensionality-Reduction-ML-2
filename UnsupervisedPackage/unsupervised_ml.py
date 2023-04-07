import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MyUnsupervisedPkg(BaseEstimator, TransformerMixin):
    def __init__(self, hyperparam1=1, hyperparam2=2):
        self.hyperparam1 = hyperparam1
        self.hyperparam2 = hyperparam2

    def fit(self, X):
        # Do something with X
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        # Do something with X
        return X
