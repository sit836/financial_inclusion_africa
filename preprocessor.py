import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Preprocessor:
    def __init__(self, num_feas, cat_feas):
        self.num_feas = num_feas
        self.cat_feas = cat_feas

        self.is_enc_fitted = False
        self.enc = OneHotEncoder(handle_unknown='ignore')

    def _enc_cat_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_enc_fitted:
            self.enc.fit_transform(X[self.cat_feas]).todense()
            self.is_enc_fitted = True
        else:
            self.enc.transform(X[self.cat_feas]).todense()
        return pd.DataFrame(np.hstack((X[self.num_feas], self.enc.fit_transform(X[self.cat_feas]).todense())),
                            columns=self.num_feas + list(self.enc.get_feature_names_out()))

    def preprocess(self, X: pd.DataFrame) ->  pd.DataFrame:
        X_processed = self._enc_cat_features(X)
        return X_processed
