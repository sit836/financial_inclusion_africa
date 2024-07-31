import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

IN_PATH = 'D:/py_projects/Financial_Inclusion_Africa/data/'
TARGET = 'bank_account'
NUM_FEAS = ['year', 'household_size', 'age_of_respondent']
# CAT_FEAS = ['country', 'uniqueid', 'location_type', 'cellphone_access',
#             'gender_of_respondent', 'relationship_with_head', 'marital_status',
#             'education_level', 'job_type']
CAT_FEAS = ['country', 'location_type', 'cellphone_access',
            'gender_of_respondent', 'relationship_with_head', 'marital_status',
            'education_level', 'job_type']

is_local_experiment = True
df_train = pd.read_csv(os.path.join(IN_PATH, 'Train.csv'))
df_test = pd.read_csv(os.path.join(IN_PATH, 'Test.csv'))
df_train[TARGET] = df_train[TARGET].map({'Yes': 1, 'No': 0})
df_submission = pd.read_csv(os.path.join(IN_PATH, 'SampleSubmission.csv'))

# print(len(set(df_train['uniqueid']).intersection(df_test['uniqueid'])))
X_train_raw, y_train_raw = df_train[NUM_FEAS + CAT_FEAS], df_train[TARGET]
print(f'df_train.shape, df_test.shape: {df_train.shape, df_test.shape}')

if is_local_experiment:
    X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw, stratify=y_train_raw,
                                                      test_size=0.3, random_state=42)

    enc = OneHotEncoder(handle_unknown='ignore')
    X_processed_train = pd.DataFrame(np.hstack((X_train[NUM_FEAS], enc.fit_transform(X_train[CAT_FEAS]).todense())),
                                     columns=NUM_FEAS + list(enc.get_feature_names_out()))
    X_processed_val = pd.DataFrame(np.hstack((X_val[NUM_FEAS], enc.transform(X_val[CAT_FEAS]).todense())),
                                   columns=NUM_FEAS + list(enc.get_feature_names_out()))
    print(X_processed_train.shape, X_processed_val.shape)

    # TODO: std
    model = LogisticRegression(random_state=0)
    model.fit(X_processed_train, y_train)
    pred_train = model.predict(X_processed_train)
    pred_val = model.predict(X_processed_val)

    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)
    print(f'mae_train: {round(mae_train, 4)}')
    print(f'mae_val: {round(mae_val, 4)}')
else:
    # train a model on all the training data
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit_transform(X_train_raw[CAT_FEAS])
    fea_names = NUM_FEAS + list(enc.get_feature_names_out())
    X_processed_train = pd.DataFrame(np.hstack((X_train_raw[NUM_FEAS], enc.transform(X_train_raw[CAT_FEAS]).todense())),
                                     columns=fea_names)
    X_processed_test = pd.DataFrame(np.hstack((df_test[NUM_FEAS], enc.fit_transform(df_test[CAT_FEAS]).todense())),
                                    columns=fea_names)

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
    df_my_submission.to_csv('./logistic_submission.csv', index=False)
