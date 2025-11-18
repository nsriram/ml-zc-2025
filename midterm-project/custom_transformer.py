import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer: set ph==0 to NaN
class PhZeroToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, ph_col="ph"):
        self.ph_col = ph_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xc = X.copy()
        if self.ph_col in Xc.columns:
            Xc.loc[Xc[self.ph_col] == 0, self.ph_col] = np.nan
        return Xc
