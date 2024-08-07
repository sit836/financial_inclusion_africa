import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from constants import IN_PATH, TARGET, NUM_FEAS, CAT_FEAS
from preprocessor import Preprocessor

is_local_experiment = False
df_train = pd.read_csv(os.path.join(IN_PATH, 'Train.csv'))
df_test = pd.read_csv(os.path.join(IN_PATH, 'Test.csv'))
df_train[TARGET] = df_train[TARGET].map({'Yes': 1, 'No': 0})
df_submission = pd.read_csv(os.path.join(IN_PATH, 'SampleSubmission.csv'))

# print(len(set(df_train['uniqueid']).intersection(df_test['uniqueid'])))
X_train_raw, y_train_raw = df_train[NUM_FEAS + CAT_FEAS], df_train[TARGET]
print(f'df_train.shape, df_test.shape: {df_train.shape, df_test.shape}')
# df_train.shape, df_test.shape: ((23524, 13), (10086, 12))

print(f'y_train_raw.value_counts(): {y_train_raw.value_counts()}')
# 0    20212
# 1     3312

if is_local_experiment:
    X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw, stratify=y_train_raw,
                                                      test_size=0.3, random_state=42)

    prep = Preprocessor(NUM_FEAS, CAT_FEAS)
    X_processed_train = prep.preprocess(X_train)
    X_processed_val = prep.preprocess(X_val)

    model = LogisticRegression(random_state=0, C=np.inf)
    model.fit(X_processed_train, y_train)
    pred_train = model.predict(X_processed_train)
    pred_val = model.predict(X_processed_val)

    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)
    print(f'mae_train: {round(mae_train, 4)}')
    print(f'mae_val: {round(mae_val, 4)}')
    # mae_train: 0.11
    # mae_val: 0.11
else:
    # train a model on all the training data
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit_transform(X_train_raw[CAT_FEAS])
    fea_names = NUM_FEAS + list(enc.get_feature_names_out())

    prep = Preprocessor(NUM_FEAS, CAT_FEAS)
    X_processed_train = prep.preprocess(X_train_raw)
    X_processed_test = prep.preprocess(df_test)

    model = LogisticRegression(random_state=0)
    model.fit(X_processed_train, y_train_raw)
    pred_test = model.predict(X_processed_test)

    df_result_train = pd.DataFrame({'unique_id': df_train['uniqueid'] + ' x ' + df_train['country'],
                                    TARGET: y_train_raw,
                                    })
    df_result_test = pd.DataFrame({'unique_id': df_test['uniqueid'] + ' x ' + df_test['country'],
                                   TARGET: pred_test,
                                   })
    df_result = pd.concat([df_result_train, df_result_test], ignore_index=True)
    df_my_submission = df_submission[['unique_id']].merge(df_result, on='unique_id')
    # df_my_submission.to_csv('./logistic_submission.csv', index=False)
