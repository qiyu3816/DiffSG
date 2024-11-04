import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import random

def MinMaxNorm(X, lower=0.0, upper=1.0):
    """
    Normalize X based on min and max.
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    return lower + (upper - lower) * (X - X_min) / (X_max - X_min)

def MeanNorm(X):
    """
    Normalize X based on mean and std.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    return (X - X_mean) / X_std

def data_preprocess_co(X):
    """
    Simplify the CO dataset.
    :param X: (sample_num, node_num * special_feature_num + common_feature_num),
                    special_feature_num=6, common_feature_num=7.
    :return: X_simplified (sample_num, node_num * special_feature_num + common_feature_num),
             special_feature_num=3, common_feature_num=0.
    """
    node_num = (X.shape[1] - 7) // 6
    X_simplified = np.zeros(shape=(X.shape[0], node_num * 3))
    # local_cost offload_transition_cost ideal_offload_execution_cost
    sum_P_t_h = np.zeros_like(X[:, 0])
    for i in range(node_num):
        sum_P_t_h = sum_P_t_h + X[:, -5] * (X[:, 6 * i + 3] ** 2)
    for i in range(node_num):
        sinr = X[:, -5] * (X[:, 6 * i + 3] ** 2) / (X[:, -1] + sum_P_t_h)
        r_u = X[:, -2] * np.log2(1.0 + sinr)

        X_simplified[:, 3 * i] = X[:, 6 * i + 4] * X[:, 6 * i + 1] / X[:, 6 * i + 2] \
                                 + (1.0 - X[:, 6 * i + 4]) * X[:, -6] * (X[:, 6 * i + 2] ** 2) * X[:, 6 * i + 1]
        X_simplified[:, 3 * i + 1] = X[:, 6 * i + 4] * X[:, 6 * i] / r_u \
                                     + (1.0 - X[:, 6 * i + 4]) * X[:, -5] * X[:, 6 * i] / r_u
        X_simplified[:, 3 * i + 2] = X[:, 6 * i + 4] * X[:, 6 * i + 1] / X[:, -7] \
                                     + (1.0 - X[:, 6 * i + 4]) * X[:, -4] * X[:, 6 * i + 1] / X[:, -7]

    return X_simplified

def read_dataset(filepath, scaler_lower_bound=0.1, scaler_upper_bound=1.1, test_size=0.2, debug=False):
    """
    Read the dataset from the specified file, automatically infer the mu_num corresponding
    to the dataset and perform the training/test set partitioning.
    :param filepath: dataset path
    :param scaler_lower_bound: scaling lower bound
    :param scaler_upper_bound: scaling upper bound
    :param test_size: testset ratio
    :param debug: debug for message print
    :return: scaled X_train, scaled X_test, Y_train for classification task, Y_test for classification task,
            Y_train for regression task, Y_test for regression task
    """
    if debug:
        print("[read_dataset] Reading dataset from", filepath)
    data = pd.read_csv(filepath)
    data_array = np.array(data)
    mu_num = int((data_array.shape[1] - 1) / 7)

    X = data_array[:, 0:-(mu_num + 1)]
    Y = np.atleast_2d(data_array[:, -(mu_num + 1):])

    scaler = MinMaxScaler(feature_range=(scaler_lower_bound, scaler_upper_bound))
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=test_size)
    Y_train_class = np.atleast_2d(Y_train[:, -(mu_num + 1)]).T
    Y_test_class = np.atleast_2d(Y_test[:, -(mu_num + 1)]).T
    Y_train_reg = np.atleast_2d(Y_train[:, -mu_num:])
    Y_test_reg = np.atleast_2d(Y_test[:, -mu_num:])

    if debug:
        print("[read_dataset] Read finished, mu_num={}, sample num={}, return.".format(mu_num, X.shape[0]))

    return X_train, X_test, Y_train_class, Y_train_reg, Y_test_class, Y_test_reg