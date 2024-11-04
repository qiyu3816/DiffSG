"""
Baseline APMonitor Optimization Suite - GEKKO, which is tested on CO, MSR and NU.
"""
import torch
from tqdm import tqdm
import numpy as np
from gekko import GEKKO

from ddpm_opt.classifier_free_CO import co_data_load, cost_calc
from ddpm_opt.classifier_free_MSR import msr_data_load
from ddpm_opt.classifier_free_NU import nu_data_load, rate_calc


def co_solve(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    m = GEKKO()  # Initialize gekko
    m.options.SOLVER = 1  # APOPT is an MINLP solver

    # optional solver settings with APOPT
    m.solver_options = ['minlp_maximum_iterations 500',  # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 10',  # treat minlp as nlp
                        'minlp_as_nlp 0',  # nlp sub-problem max iterations
                        'nlp_maximum_iterations 50',  # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1',  # maximum deviation from whole number
                        'minlp_integer_tol 0.05',  # convergence tolerance
                        'minlp_gap_tol 0.01']

    y1 = m.Var(value=0.3, lb=0, ub=1)
    y2 = m.Var(value=0.35, lb=0, ub=1)
    y3 = m.Var(value=0.35, lb=0, ub=1)
    y4 = m.Var(value=0, lb=0, ub=1, integer=True)
    y5 = m.Var(value=0, lb=0, ub=1, integer=True)
    y6 = m.Var(value=0, lb=0, ub=1, integer=True)
    m.Equation(y1 * y4 + y2 * y5 + y3 * y6 <= 1)
    m.Obj((1 - y4) * x1 + y4 * (x2 + x3 / y1) + (1 - y5) * x4 + y5 * (x5 + x6 / y2) + (1 - y6) * x7 + y6 * (x8 + x9 / y3))
    m.solve(disp=False)

    f1 = float(y1.VALUE[0]) * float(y4.VALUE[0])
    f2 = float(y2.VALUE[0]) * float(y5.VALUE[0])
    f3 = float(y3.VALUE[0]) * float(y6.VALUE[0])
    return f1, f2, f3

def sBB_co():
    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test = X_test * (scaler_max - scaler_min) + scaler_min
    node_num = Y_train.shape[1]

    Y_pred = None
    used_sample_num = 100
    for i in tqdm(range(used_sample_num)):
        if Y_pred is None:
            y1, y2, y3 = co_solve(X_test[i, 0], X_test[i, 1], X_test[i, 2],
                                  X_test[i, 3], X_test[i, 4], X_test[i, 5],
                                  X_test[i, 6], X_test[i, 7], X_test[i, 8])
            Y_pred = np.atleast_2d(np.array([y1, y2, y3]))
        else:
            y1, y2, y3 = co_solve(X_test[i, 0], X_test[i, 1], X_test[i, 2],
                                  X_test[i, 3], X_test[i, 4], X_test[i, 5],
                                  X_test[i, 6], X_test[i, 7], X_test[i, 8])
            Y_pred = np.concatenate((Y_pred, np.atleast_2d(np.array([y1, y2, y3]))))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    Y_pred_tensor = torch.tensor(Y_pred, dtype=torch.float32)

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


def msr_solve(X, M, W):
    m = GEKKO()
    m.options.SOLVER = 3
    m.solver_options = ['linear_solver ma97']

    ps = m.Array(m.Var, (M))
    for i in range(M):
        ps[i].value = W / M
        ps[i].lower = 0.01
        ps[i].upper = W - (M - 1) * 0.01

    m.Equation(m.sum([ps[i] for i in range(M)]) == W)
    m.Obj(-m.sum([m.log(1 + X[i] * ps[i]) / m.log(2) for i in range(M)]))
    m.solve(disp=False)
    p = np.array([ps[i] for i in range(M)])
    return p

def sBB_msr():
    dataset_path = "../datasets/80c_20w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    M, W = custom_config['M'], custom_config['W']
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test = X_test * (scaler_max - scaler_min) + scaler_min

    Y_pred = None
    used_sample_num = 3
    for i in tqdm(range(used_sample_num)):
        if Y_pred is None:
            Y_pred = np.atleast_2d(msr_solve(X_test[i], M, W)).T
        else:
            Y_pred = np.concatenate((Y_pred, np.atleast_2d(msr_solve(X_test[i], M, W)).T))

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


def nu_solve(x1, y1, x2, y2, x3, y3):
    m = GEKKO()
    m.options.SOLVER = 3
    m.solver_options = ['linear_solver ma97']
    u1 = m.Var(value=200, lb=-200, ub=600)
    u2 = m.Var(value=200, lb=-200, ub=600)
    p1 = m.Var(value=6, lb=0.1, ub=17.8)
    p2 = m.Var(value=6, lb=0.1, ub=17.8)
    p3 = m.Var(value=6, lb=0.1, ub=17.8)
    h1 = m.sqrt(60 / (22500 + (u1 - x1) ** 2 + (u2 - y1) ** 2))
    h2 = m.sqrt(60 / (22500 + (u1 - x2) ** 2 + (u2 - y2) ** 2))
    h3 = m.sqrt(60 / (22500 + (u1 - x3) ** 2 + (u2 - y3) ** 2))
    sinr1 = p1 / (m.if2(p2 - p1, 1, 0) * p2 + m.if2(p3 - p1, 1, 0) * p3 + 110 / (h1 ** 2))
    sinr2 = p2 / (m.if2(p1 - p2, 1, 0) * p1 + m.if2(p3 - p2, 1, 0) * p3 + 110 / (h2 ** 2))
    sinr3 = p3 / (m.if2(p1 - p3, 1, 0) * p1 + m.if2(p2 - p3, 1, 0) * p2 + 110 / (h3 ** 2))

    m.Equation(p1 + p2 + p3 == 18)
    m.Equation((h1 - h2) * (p2 - p1) >= 0)
    m.Equation((h1 - h3) * (p3 - p1) >= 0)
    m.Equation((h3 - h2) * (p2 - p3) >= 0)
    m.Obj(-(m.log(1 + sinr1) / m.log(2) + m.log(1 + sinr2) / m.log(2) + m.log(1 + sinr3) / m.log(2)))
    m.solve(disp=False)
    return u1.VALUE[0], u2.VALUE[0], p1.VALUE[0], p2.VALUE[0], p3.VALUE[0]

def sBB_nu():
    width, height = 400, 400
    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    K, P_sum = custom_config['K'], custom_config['P_sum']

    Y_pred = None
    used_sample_num = 20
    for i in tqdm(range(used_sample_num)):
        if Y_pred is None:
            u1, u2, p1, p2, p3 = nu_solve(X_test[i, 0], X_test[i, 1], X_test[i, 2],
                                          X_test[i, 3], X_test[i, 4], X_test[i, 5])
            Y_pred = np.atleast_2d(np.array([u1, u2, p1, p2, p3]))
        else:
            u1, u2, p1, p2, p3 = nu_solve(X_test[i, 0], X_test[i, 1], X_test[i, 2],
                                          X_test[i, 3], X_test[i, 4], X_test[i, 5])
            Y_pred = np.concatenate((Y_pred, np.atleast_2d(np.array([u1, u2, p1, p2, p3]))))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    for i in range(K):
        X_test_tensor[:, 2 * i] *= width
        X_test_tensor[:, 2 * i + 1] *= height
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    Y_test_tensor[:, 0] *= width
    Y_test_tensor[:, 1] *= height
    Y_test_tensor[:, 2:] *= P_sum

    Y_pred_tensor = torch.tensor(Y_pred, dtype=torch.float32)
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
    # sBB_co()
    # exceeded_ratio=1.005359411239624 8.96s/it

    sBB_msr()
    # 3c less_ratio=1.0000000003410903, 8.63s/it
    # 8c less_ratio=1.0037997606603422, 25.02s/it

    # sBB_nu()
    # less_ratio=0.485004186630249, 9.59s/it
