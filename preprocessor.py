import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Preprocessor:
    def __init__(self, num_feas, cat_feas):
        self.num_feas = num_feas
        self.cat_feas = cat_feas
        # self.created_feas = ['country_latitude', 'country_longitude']
        self.created_feas = []

        self.is_enc_fitted = False
        self.enc = OneHotEncoder(handle_unknown='ignore')

    def _enc_cat_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_enc_fitted:
            self.enc.fit_transform(X[self.cat_feas]).todense()
            self.is_enc_fitted = True
        else:
            self.enc.transform(X[self.cat_feas]).todense()
        return pd.DataFrame(
            np.hstack((X[self.num_feas + self.created_feas], self.enc.fit_transform(X[self.cat_feas]).todense())),
            columns=self.num_feas + self.created_feas + list(self.enc.get_feature_names_out()))

    def _enc_geo_info(self, X: pd.DataFrame) -> pd.DataFrame:
        X['country_latitude'] = X['country'].map({'Kenya': 0.02, 'Rwanda': 1.94, 'Tanzania': 6.37, 'Uganda': 1.37})
        X['country_longitude'] = X['country'].map(
            {'Kenya': 37.91, 'Rwanda': 29.87, 'Tanzania': 34.89, 'Uganda': 32.29})
        return X

    def _order_enc(self, X: pd.DataFrame) -> pd.DataFrame:
        X['education_level_num'] = X['education_level'].map({'Other/Dont know/RTA': -1,
                                                             'No formal education': 0,
                                                             'Primary education': 1,
                                                             'Secondary education': 2,
                                                             'Vocational/Specialised training': 3,
                                                             'Tertiary education': 4,
                                                             })
        return X

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X
        X_processed = self._order_enc(X_processed)
        X_processed = self._enc_cat_features(X_processed)
        return X_processed
