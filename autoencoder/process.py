import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.preprocessing import MinMaxScaler

def load_data():
    train_set = '../data/UNSW_nB15-training-set.csv'
    test_set = '../data/UNSW_nB15-testing-set.csv'

    train = pd.read_csv(train_set, index_col='id')
    test = pd.read_csv(test_set, index_col='id')

    training_label = train['label'].values
    testing_label = test['label'].values

    temp_train = training_label
    temp_test = testing_label

    unsw = pd.concat([train, test])
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])

    unsw.drop(['label', 'attack_cat'], axis=1, inplace=True)
    unsw_value = unsw.values

    scaler = MinMaxScaler(feature_range=(0,1))
    unsw_value = scaler.fit_transform(unsw_value)
    train_set = unsw_value[:len(train), :]
    test_set = unsw_value[len(train):, :]

    return train_set, temp_train, test_set, temp_test