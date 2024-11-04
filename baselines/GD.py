"""
Vanilla gradient descent.
"""
import numpy as np
import torch
from tqdm import tqdm

from ddpm_opt.classifier_free_CO import co_data_load, cost_calc
from ddpm_opt.classifier_free_MSR import msr_data_load
from ddpm_opt.classifier_free_NU import nu_data_load, rate_calc

def co_gradient(x, y, node_num, lambda1, lambda2):
    """
    Calculate the gradient of the objective function with Lagrange functions.
    :param y: Decision and allocation.
    """
    gradient = np.zeros_like(y)
    for i in range(node_num):
        gradient[:, i] = -x[:, 3 * i] + x[:, 3 * i + 1] + x[:, 3 * i + 2] / y[:, i + node_num] + (1 - 2 * y[:, i]) * lambda1
        gradient[:, i + node_num] = - x[:, 3 * i + 2] / (y[:, i + node_num] ** 2) * y[:, i] + (np.sum(y[:, -node_num:], axis=1) * 2 - 1) * lambda2
    return gradient

def co_solve():
    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test = X_test * (scaler_max - scaler_min) + scaler_min
    node_num = Y_train.shape[1]

    used_sample_num = 10000
    iteration = 100
    Y_pred = np.ones((used_sample_num, 2 * node_num))
    Y_pred[:, -node_num:] = 1 / node_num
    for i in tqdm(range(iteration)):
        grad = co_gradient(X_test[:used_sample_num, :], Y_pred, node_num, 1.0, 1.0)
        Y_pred -= grad * 0.1

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    Y_pred_tensor = torch.tensor(Y_pred[:, -node_num:], dtype=torch.float32)

    # normalization because the cost_calc cannot cope with the extremely invalid solutions
    min_values, _ = torch.min(Y_pred_tensor, dim=1, keepdim=True)
    max_values, _ = torch.max(Y_pred_tensor, dim=1, keepdim=True)
    Y_pred_tensor = (Y_pred_tensor - min_values) / (max_values - min_values)
    # Y_pred_tensor = torch.softmax(Y_pred_tensor, dim=1)

    pred_cost = cost_calc(X_test_tensor[:used_sample_num], Y_pred_tensor)
    true_cost = cost_calc(X_test_tensor[:used_sample_num], Y_test_tensor[:used_sample_num])

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_cost:\n", pred_cost[:20])
    print("true_cost:\n", true_cost[:20])
    print(f"exceeded ratio: {torch.sum(pred_cost) / torch.sum(true_cost)}")
    print("avg cost diff:\n", torch.mean(pred_cost - true_cost))


def msr_gradient(gs, schemes):
    """
    :param gs: (sample_num, M)
    :param schemes: (sample_num, M)
    :return: schemes' grad (sample_num, M)
    """
    product = gs * schemes
    grad = gs / ((product + 1.0) * np.log(2)) - 1.0 / np.atleast_2d(((np.sum(schemes, axis=1) - 1) ** 2)).T
    return grad

def msr_solve():
    dataset_path = "../datasets/80c_20w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    M, W = custom_config['M'], custom_config['W']
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test = X_test * (scaler_max - scaler_min) + scaler_min

    used_sample_num = 1000
    Y_pred = np.ones_like(Y_test[:used_sample_num, :]) / M * W
    iteration = 100
    for i in tqdm(range(iteration)):
        Y_pred += msr_gradient(X_test[:used_sample_num, :], Y_pred[:used_sample_num]) * 0.001

    Y_pred_sum = np.atleast_2d(np.sum(Y_pred, axis=1)).T
    Y_pred += (W - Y_pred_sum) / M
    pred_rate = np.sum(np.log2(1.0 + Y_pred * X_test[:used_sample_num]), axis=1)
    true_rate = np.sum(np.log2(1.0 + Y_test[:used_sample_num] * X_test[:used_sample_num]), axis=1)

    np.set_printoptions(precision=8, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test[:20])
    print("pred_cost:\n", pred_rate[:20])
    print("true_cost:\n", true_rate[:20])
    print(f"less ratio: {np.sum(pred_rate) / np.sum(true_rate)}")
    print("avg rate diff:\n", np.mean(pred_rate - true_rate))


def nu_gradient(cur_y, coordinates, K):
    grad = np.zeros_like(cur_y)
    d1_square = (cur_y[:, 0] - coordinates[:, 0]) ** 2 + (cur_y[:, 1] - coordinates[:, 1]) ** 2
    d2_square = (cur_y[:, 0] - coordinates[:, 2]) ** 2 + (cur_y[:, 1] - coordinates[:, 3]) ** 2
    d3_square = (cur_y[:, 0] - coordinates[:, 4]) ** 2 + (cur_y[:, 1] - coordinates[:, 5]) ** 2
    for i in range(K):
        if i == 0:
            tmp = 6 + 11 / 6 * (22500 + d1_square)
        if i == 1:
            tmp = 6 + 11 / 6 * (22500 + d2_square)
        if i == 2:
            tmp = 6 + 11 / 6 * (22500 + d3_square)
        grad[:, 0] += - cur_y[:, 2 + i] * (cur_y[:, 0] - coordinates[:, 2 * i]) * 11 / 3 / (tmp ** 2) / (1 + cur_y[:, 2 + i] / tmp) / np.log(2) \
                      + 2 * (coordinates[:, 2 * i + 1] - cur_y[:, 0]) / ((d1_square + d2_square + d3_square) ** 2)
        grad[:, 1] += - cur_y[:, 2 + i] * (cur_y[:, 1] - coordinates[:, 2 * i + 1]) * 11 / 3 / (tmp ** 2) / (1 + cur_y[:, 2 + i] / tmp) / np.log(2) \
                      + 2 * (coordinates[:, 2 * i + 1] - cur_y[:, 1]) / ((d1_square + d2_square + d3_square) ** 2)
        grad[:, 2 + i] = - 1 / tmp / (1 + cur_y[:, 2 + i] / tmp) / np.log(2) + 1 / ((np.sum(cur_y[:, 2:], axis=1) - 18) ** 2)
    return grad


def nu_solve():
    width, height = 400, 400
    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    K, P_sum = custom_config['K'], custom_config['P_sum']

    used_sample_num = 3000
    Y_pred = np.ones_like(Y_test[:used_sample_num, :]) * P_sum / K - 0.01
    Y_pred[:, 0], Y_pred[:, 1] = width / 2, height / 2
    iterations = 100
    for i in tqdm(range(iterations)):
        grad = nu_gradient(Y_pred, X_test[:used_sample_num, :], K)
        Y_pred += grad * 0.1

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    for i in range(K):
        X_test_tensor[:, 2 * i] *= width
        X_test_tensor[:, 2 * i + 1] *= height
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    Y_test_tensor[:, 0] *= width
    Y_test_tensor[:, 1] *= height
    Y_test_tensor[:, 2:] *= P_sum

    Y_pred_tensor = torch.tensor(Y_pred, dtype=torch.float32)
    Y_pred_sum = torch.sum(Y_pred_tensor[:, -K:], dim=1).unsqueeze(dim=1)
    Y_pred_tensor[:, -K:] = Y_pred_tensor[:, -K:] / Y_pred_sum * P_sum
    pred_rate = rate_calc(Y_pred_tensor, X_test_tensor[:used_sample_num])
    true_rate = rate_calc(Y_test_tensor[:used_sample_num], X_test_tensor[:used_sample_num])

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)
    print("Y_pred:\n", Y_pred_tensor[:20])
    print("Y_test:\n", Y_test_tensor[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_cost:\n", pred_rate[:20])
    print("true_cost:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))


if __name__ == "__main__":
    # co_solve()

    # msr_solve()

    nu_solve()
