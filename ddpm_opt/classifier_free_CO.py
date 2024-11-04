"""
The Classifier-Free Guidance version of DIFFSG specified for Computation Offloading,
re-structured from the rough version of DDPM.
"""

import random
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import math
import time
import datetime
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from ddpm_opt.diffusion import generate_cosine_schedule, init_weights
from ddpm_opt.UNetCF import UNet1D
from ddpm_opt.ema import ExponentialMovingAverage
from utils.dataset import data_preprocess_co


def condition_C(y, x, x_scaler_min, x_scaler_max):
    """
    Calculate the customized objectives and constraints in numerical conditions.
    :param y: (batch_size, node_num)
    :param x: (batch_size, sfn * node_num)
    :return: (batch_size, sfn * node_num + c_dim)
    """
    y_norm = torch.softmax(y, dim=1) + 0.000001

    D = torch.where(y_norm > 0.1, 1, 0)
    x_src = (x - x_scaler_min) * (x_scaler_max - x_scaler_min) + x_scaler_min
    local_cost = torch.index_select(x_src, 1, torch.tensor([i for i in range(0, x.shape[1], 3)], device=x.device))
    offload_transition_cost = torch.index_select(x_src, 1, torch.tensor([i for i in range(1, x.shape[1], 3)], device=x.device))
    ideal_offload_execution_cost = torch.index_select(x_src, 1, torch.tensor([i for i in range(2, x.shape[1], 3)], device=x.device))

    total_cost = torch.sum((1 - D) * local_cost + D * (offload_transition_cost + ideal_offload_execution_cost / y_norm), dim=1)[:, None]
    total_cost /= 10
    x = torch.cat((x, total_cost), dim=1)
    return x

# DDPM
class DDPM(nn.Module):
    """
    DDPM in the version of "Classifier-Free Diffusion Guidance".
    """

    def __init__(self,
                 T,
                 model,
                 node_num,
                 alphas,
                 device,
                 data_size,
                 custom_config=None,
                 uncond_prob=0.1,
                 ema_decay=0.9999,
                 ema_start=1000,
                 ema_update_rate=5,
                 debug=False):
        super(DDPM, self).__init__()
        self.T = T
        self.model = model
        self.node_num = node_num
        self.data_size = data_size
        self.custom_config = custom_config
        self.debug = debug
        self.device = device

        self.uncond_prob = uncond_prob

        betas = 1.0 - alphas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sqrt_betas", to_torch(np.sqrt(betas)))

        self.ema = ExponentialMovingAverage(self.model, ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate

    def forward(self, y, cond):
        ts = torch.randint(low=0, high=self.T, size=(1, y.shape[0]), device=self.device)
        noise = torch.randn_like(y, device=self.device)
        y_t = self.sqrt_alphas_cumprod[ts, None] * y + self.sqrt_one_minus_alphas_cumprod[ts, None] * noise
        y_t = torch.squeeze(y_t)
        # cond = condition_C(y_t, cond, self.custom_config['scaler_min'], self.custom_config['scaler_max'])

        cond_mask = torch.bernoulli(torch.fill(torch.zeros(cond.shape[0], device=self.device), 1 - self.uncond_prob))[:, None]

        estimated_noise = self.model(y_t, ts / self.T, cond, cond_mask)
        if random.random() < 0.01:
            print(f"noise {noise[1, :]}, estimated_noise {estimated_noise[1, :]}")
        return F.mse_loss(noise, estimated_noise)

    def sample(self, cond, omega=1.0):
        y_t = torch.randn(cond.shape[0], *self.data_size).to(self.device)
        y_t = torch.squeeze(y_t)

        # cond = condition_C(y_t, cond, self.custom_config['scaler_min'], self.custom_config['scaler_max'])
        cond_mask_0 = torch.zeros(cond.shape[0], device=self.device)[:, None]
        cond_mask_1 = torch.ones(cond.shape[0], device=self.device)[:, None]

        y_i_record = []  # keep track of generated steps in case want to plot something
        eps_i_record = []
        for i in range(self.T - 1, -1, -1):
            # cond = condition_C(y_t, cond[:, :-1], self.custom_config['scaler_min'], self.custom_config['scaler_max'])  # cdim
            eps_0 = self.model(y_t, torch.full(size=(1, cond.shape[0]), fill_value=i, device=self.device) / self.T, cond, cond_mask_0)
            eps_1 = self.model(y_t, torch.full(size=(1, cond.shape[0]), fill_value=i, device=self.device) / self.T, cond, cond_mask_1)

            noise = torch.randn(cond.shape[0], *self.data_size).to(self.device) if i > 1 else 0
            noise = torch.squeeze(noise) if i > 1 else 0

            eps = (1 + omega) * eps_1 - omega * eps_0
            y_t = (y_t - self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i] * eps) * self.reciprocal_sqrt_alphas[i] \
                  + (1.0 - self.alphas_cumprod[i - 1 if i - 1 >= 0 else 0]) / (1.0 - self.alphas_cumprod[i]) * noise

            if i > self.T - 5:  # Normalization in case that solution value explosion in the early sampling.
                y_t = (y_t - torch.mean(y_t)) / torch.sqrt(torch.var(y_t))

            if i % 20 == 0 or i == self.T - 1 or i < 5 or self.T <= 20:
                y_i_record.append(y_t.detach().cpu().numpy())
                eps_i_record.append(eps.detach().cpu().numpy())

        self.y_i_record = np.array(y_i_record)
        self.eps_i_record = np.array(eps_i_record)
        return y_t


# Dataset and preprocessing
def co_data_load(dataset_path):
    """
    Load, scale and split CO dataset.
    """
    shuffle = False
    train_ratio, test_ratio = 0.7, 0.3
    C_dim = 1  # only objective function value

    src_csv = pd.read_csv(dataset_path, header=None)
    src_data = np.array(src_csv)
    if shuffle:
        np.random.shuffle(src_data)
    node_num = (src_data.shape[1] - 1) // 7
    X, Y = src_data[:, :6 * node_num], src_data[:, -node_num:]

    # special_feature_num, common_feature_num = 6, 7
    F_t = 2.5e9
    kappa = 1e-28
    Pt = 0.3
    PI = 0.1
    theta = 1.0
    B = 10e5
    N0 = 7.96159e-13
    common_features = np.expand_dims(np.array([F_t, kappa, Pt, PI, theta, B, N0], dtype=float), axis=0)
    X = np.concatenate((X, np.tile(common_features, (X.shape[0], 1))), axis=1)
    X = data_preprocess_co(X)

    # de-abnormal
    indices = []
    for i in range(X.shape[0]):
        if np.all(np.where(X[i, :] < 10.0, 1, 0)):
            indices.append(i)
    X, Y = X[indices, :], Y[indices, :]
    scaler_min, scaler_max = np.min(X), np.max(X)

    X = (X - scaler_min) / (scaler_max - scaler_min)
    special_feature_num, common_feature_num = 3, 0
    custom_config = {'sfn': special_feature_num, 'cfn': common_feature_num, 'cdim': C_dim,
                     'scaler_min': scaler_min, 'scaler_max': scaler_max}

    X_train, Y_train = X[:int(src_data.shape[0] * train_ratio)], Y[:int(src_data.shape[0] * train_ratio)]
    X_test, Y_test = X[-int(src_data.shape[0] * test_ratio):], Y[-int(src_data.shape[0] * test_ratio):]
    return X_train, Y_train, X_test, Y_test, custom_config


def train_ddpm_co():
    epochs = 200
    T = 20
    use_ema = False
    warmup_epoch = 5

    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    node_num = Y_train.shape[1]

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=node_num, proj_dim=64, cond_dim=custom_config['sfn'] * node_num,# + custom_config['cdim'],
                   dims=(64, 32, 16, 8), is_attn=(False, False, False, False), middle_attn=False, n_blocks=3)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, node_num, alphas, device, (1, node_num), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.apply(init_weights)
    diffusion_model.to(device)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 80, 150])

    ema_step_cnt = 1
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_sample_num = 0
        for x, y_true in tqdm(data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            loss = diffusion_model(y_true, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (use_ema and epoch > warmup_epoch
                    and ema_step_cnt > diffusion_model.ema_start and ema_step_cnt % diffusion_model.ema_update_rate == 0):
                diffusion_model.ema.update_parameters(diffusion_model.model)
            epoch_loss += loss.item()
            epoch_sample_num += x.shape[0]
            ema_step_cnt += 1
        print(f"Epoch: {epoch}, Loss: {epoch_loss / epoch_sample_num}")
        lr_scheduler.step()

    return diffusion_model


def cost_calc(X, Y):
    """
    Calculate the overall cost of each sample with the specified X and predicted Y. Totally local is acceptable.
    :param X: The simplified X. (sample_num, node_num * special_feature_num + common_feature_num)
    :param Y: The resource allocation results. (sample_num, node_num)
    :return: The overall cost of the strategy. (sample_num)
    """
    D = torch.where(Y > 0.1, 1, 0)

    Y = torch.where(D == 1, Y, 0)
    Y_sum = torch.sum(Y, dim=1)
    D_sum = torch.sum(D, dim=1)
    D_sum = torch.where(D_sum == 0, 0.00001, D_sum)
    Y_diff = torch.atleast_2d((1 - Y_sum) / D_sum).T
    Y_diff = torch.cat((Y_diff, Y_diff, Y_diff), dim=1)
    Y = torch.where(D == 1, Y + Y_diff, 0.00001)

    local_cost = torch.index_select(X, 1, torch.tensor([i for i in range(0, X.shape[1], 3)], device=X.device))
    offload_transition_cost = torch.index_select(X, 1, torch.tensor([i for i in range(1, X.shape[1], 3)], device=X.device))
    ideal_offload_execution_cost = torch.index_select(X, 1, torch.tensor([i for i in range(2, X.shape[1], 3)], device=X.device))

    # f(D,R)  #############
    total_cost = torch.sum((1 - D) * local_cost + D * (offload_transition_cost + ideal_offload_execution_cost / Y), dim=1)
    return total_cost


def customized_real_decoder(Y_pred):
    """
    Decoder transforming the raw output of diffusion sampling into feasible solution.
    :param Y_pred: (sample_num, mu_num)
    :return: Y_pred_decoded(sample_num, mu_num)
    """
    Y_pred_decoded = torch.softmax(Y_pred, dim=1)
    condition = (Y_pred < -10).all(dim=1)
    Y_pred_decoded = torch.where(condition.unsqueeze(1), 0.0, Y_pred_decoded)
    return Y_pred_decoded


@torch.no_grad()
def load_test_co():
    T = 20
    omega = 180.0

    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
    node_num = Y_train.shape[1]

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=node_num, proj_dim=64, cond_dim=custom_config['sfn'] * node_num,# + custom_config['cdim'],
                   dims=(64, 32, 16, 8), is_attn=(False, False, False, False), middle_attn=False, n_blocks=3)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, node_num, alphas, device, (1, node_num), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ready_models/ClassifierFree/3n_co_20240107150607.pt"))
    diffusion_model.to(device)

    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = diffusion_model.sample(x, omega)
        else:
            Y_pred = torch.cat((Y_pred, diffusion_model.sample(x, omega)))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)

    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test_tensor = X_test_tensor * (scaler_max - scaler_min) + scaler_min
    Y_pred_decoded = customized_real_decoder(Y_pred)
    pred_cost = cost_calc(X_test_tensor, Y_pred_decoded)
    true_cost = cost_calc(X_test_tensor, Y_test_tensor)

    pred_decision = torch.where(Y_pred_decoded > 0.1, 1, 0)
    true_decision = torch.where(Y_test_tensor > 0.1, 1, 0)
    pred_cls = np.zeros(X_test_tensor.shape[0], dtype=int)
    true_cls = np.zeros(X_test_tensor.shape[0], dtype=int)
    for i in range(Y_pred_decoded.shape[1]):
        pred_cls = pred_cls + (pred_decision[:, i] * (2 ** (Y_pred_decoded.shape[1] - i - 1))).to("cpu").numpy()
        true_cls = true_cls + (true_decision[:, i] * (2 ** (Y_pred_decoded.shape[1] - i - 1))).to("cpu").numpy()
    true_cls = np.where(true_cls == pred_cls, 1, 0)

    terrible_cnt = torch.where(pred_cost / true_cost > 1.2, 1, 0)
    terrible_cnt &= torch.where(pred_cost > 10.0, 1, 0)
    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred_decoded[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_cost:\n", pred_cost[:20])
    print("true_cost:\n", true_cost[:20])
    print(f"exceeded ratio: {torch.sum(pred_cost) / torch.sum(true_cost)}")
    print("avg cost diff:\n", torch.mean(pred_cost - true_cost))
    print(f"terrible samples num: {torch.sum(terrible_cnt)}/{X_test_tensor.shape[0]}.")
    print(f"accuracy: {np.sum(true_cls)}/{true_cls.shape[0]}")

@torch.no_grad()
def load_test_co_debug():
    """
    Load ready model for debug especially.
    """
    T = 400
    omega = 150

    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
    node_num = Y_train.shape[1]

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=node_num, proj_dim=64, cond_dim=custom_config['sfn'] * node_num,# + custom_config['cdim'],
                   dims=(64, 32, 16, 8), is_attn=(False, False, False, False), middle_attn=False, n_blocks=3)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, node_num, alphas, device, (1, node_num), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ready_models/ClassifierFree/3n_co_20240106162919.pt"))
    diffusion_model.to(device)

    Y_pred = None
    print_round = 1
    want2look = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = diffusion_model.sample(x, omega)
        else:
            Y_pred = torch.cat((Y_pred, diffusion_model.sample(x, omega)))
        print_round -= 1
        for i in want2look:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(x[i], Y_test[i])
            for j in range(diffusion_model.y_i_record.shape[0]):
                print(diffusion_model.y_i_record[j, i, :], diffusion_model.eps_i_record[j, i, :])
        if print_round == 0:
            break

    display_sampling_time = 0
    if display_sampling_time == 1:
        total_time = 0
        for i in tqdm(want2look):
            tmp = torch.tensor(X_test[i], device=device, dtype=torch.float32)
            a = time.time()
            Y_pred = diffusion_model.sample(tmp, omega)
            b = time.time()
            total_time += b - a
        print(f"Avg single sample time {total_time * 1000 / len(want2look)} ms")


def validation_data_gen():
    """
    To validate the model efficacy, generate a simple dataset.
    """
    X_base = np.random.random((1000, 3))
    X_1 = np.concatenate((X_base + 1, X_base, X_base), axis=1)
    Y_1 = np.zeros((1000, 3))
    Y_1[:, 0] = 1
    data_1 = np.concatenate((Y_1, X_1), axis=1)

    X_2 = np.concatenate((X_base, X_base + 1, X_base), axis=1)
    Y_2 = np.zeros((1000, 3))
    Y_2[:, 1] = 1
    data_2 = np.concatenate((Y_2, X_2), axis=1)

    X_3 = np.concatenate((X_base, X_base, X_base + 1), axis=1)
    Y_3 = np.zeros((1000, 3))
    Y_3[:, 2] = 1
    data_3 = np.concatenate((Y_3, X_3), axis=1)

    special_feature_num, common_feature_num = 3, 0
    custom_config = {'sfn': special_feature_num, 'cfn': common_feature_num}

    src_data = np.concatenate((data_1, data_2, data_3), axis=0)

    num_rows = src_data.shape[0]
    random_order = np.random.permutation(num_rows)
    src_data = src_data[random_order]

    X, Y = src_data[:, 3:], src_data[:, :3]
    train_ratio, test_ratio = 0.7, 0.3
    X_train, Y_train = X[:int(src_data.shape[0] * train_ratio)], Y[:int(src_data.shape[0] * train_ratio)]
    X_test, Y_test = X[-int(src_data.shape[0] * test_ratio):], Y[-int(src_data.shape[0] * test_ratio):]
    return X_train, Y_train, X_test, Y_test, custom_config

def validate_ddpm_co():
    """
    Training a model based on the validating dataset.
    """
    epochs = 500
    T = 500
    use_ema = False
    warmup_epoch = 5

    X_train, Y_train, X_test, Y_test, custom_config = validation_data_gen()
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    node_num = Y_train.shape[1]

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=node_num, proj_dim=64, cond_dim=custom_config['sfn'] * node_num,# + custom_config['cdim'],
                   dims=(32, 16, 8), is_attn=(False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, node_num, alphas, device, (1, 3), custom_config, 0.0,
                           0.9999, 10, 5, False)
    diffusion_model.apply(init_weights)
    diffusion_model.to(device)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 150, 350])

    ema_step_cnt = 1
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_sample_num = 0
        for x, y_true in tqdm(data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            loss = diffusion_model(y_true, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (use_ema and epoch > warmup_epoch
                    and ema_step_cnt > diffusion_model.ema_start and ema_step_cnt % diffusion_model.ema_update_rate == 0):
                diffusion_model.ema.update_parameters(diffusion_model.model)
            epoch_loss += loss.item()
            epoch_sample_num += x.shape[0]
            ema_step_cnt += 1
        print(f"Epoch: {epoch}, Loss: {epoch_loss / epoch_sample_num}")
        lr_scheduler.step()

    return diffusion_model

@torch.no_grad()
def test_ddpm():
    """
    Test the accuracy of the validating model.
    """
    T = 500
    omega = 30.0

    X_train, Y_train, X_test, Y_test, custom_config = validation_data_gen()
    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
    node_num = Y_train.shape[1]

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=node_num, proj_dim=64, cond_dim=custom_config['sfn'] * node_num,# + custom_config['cdim'],
                   dims=(32, 16, 8), is_attn=(False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, node_num, alphas, device, (1, 3), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ready_models/ClassifierFree/3n_co_20240106142438.pt"))
    diffusion_model.to(device)

    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = diffusion_model.sample(x, omega)
        else:
            Y_pred = torch.cat((Y_pred, diffusion_model.sample(x, omega)))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)

    Y_pred = torch.softmax(Y_pred, dim=1)

    pred_decision = torch.where(Y_pred > 0.1, 1, 0)
    true_decision = torch.where(Y_test_tensor > 0.1, 1, 0)
    pred_cls = np.zeros(X_test_tensor.shape[0], dtype=int)
    true_cls = np.zeros(X_test_tensor.shape[0], dtype=int)
    for i in range(Y_pred.shape[1]):
        pred_cls = pred_cls + (pred_decision[:, i] * (2 ** (Y_pred.shape[1] - i - 1))).to("cpu").numpy()
        true_cls = true_cls + (true_decision[:, i] * (2 ** (Y_pred.shape[1] - i - 1))).to("cpu").numpy()
    true_cls = np.where(true_cls == pred_cls, 1, 0)

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print(f"accuracy: {np.sum(true_cls)}/{true_cls.shape[0]}")


if __name__ == "__main__":
    print("########## Classifier-Free guidance diffusion for Computation Offloading. ##########")
    # diffusion_model = train_ddpm_co()
    #
    # torch.save(diffusion_model.state_dict(), f"../ready_models/ClassifierFree/{diffusion_model.node_num}n_co_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")

    load_test_co()
    #
    # load_test_co_debug()

    # diffusion_model = validate_ddpm_co()
    # torch.save(diffusion_model.state_dict(), f"../ready_models/ClassifierFree/{diffusion_model.node_num}n_co_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")
    # test_ddpm()