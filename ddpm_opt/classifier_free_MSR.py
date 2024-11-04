"""
The Classifier-Free Guidance version of DIFFSG specified for Maximum Sum Rate of Channels (MSR).
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


def condition_C(y, x, x_scaler_min, x_scaler_max):
    """
    Calculate the customized objectives and constraints in numerical conditions.
    :param y: (batch_size, channel_num)
    :param x: (batch_size, sfn * node_num)
    :return: (batch_size, sfn * node_num + c_dim)
    """
    y_norm = (y - y.min()) / (y.max() - y.min())
    y_norm = torch.softmax(y_norm, dim=1)

    x_src = (x - x_scaler_min) * (x_scaler_max - x_scaler_min) + x_scaler_min

    total_rate = torch.sum(torch.log2(1 + x_src * y_norm), dim=1)[:, None]
    x = torch.cat((x, total_rate), dim=1)
    return x


# DDPM
class DDPM(nn.Module):
    """
    DDPM in the version of "Classifier-Free Diffusion Guidance".
    """

    def __init__(self,
                 T,
                 model,
                 M,
                 W,
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
        self.M = M
        self.W = W
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
        if random.random() < 0.005:
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
def msr_data_load(dataset_path):
    """
    Load, scale and split Maximum Sum Rate of Channels dataset.
    """
    shuffle = False
    train_ratio, test_ratio = 0.7, 0.3
    C_dim = 1  # only objective function value

    src_csv = pd.read_csv(dataset_path, header=None)
    src_data = np.array(src_csv)
    if shuffle:
        np.random.shuffle(src_data)
    M = (src_data.shape[1] - 1) // 2
    W = float(dataset_path.split('_')[-2][:-1])
    X, Y = src_data[:, :M], src_data[:, -M:]

    # special_feature_num, common_feature_num = 1, 0
    scaler_min, scaler_max = np.min(X), np.max(X)
    X = (X - scaler_min) / (scaler_max - scaler_min)
    special_feature_num, common_feature_num = 1, 0
    custom_config = {'M': M, 'W': W, 'sfn': special_feature_num, 'cfn': common_feature_num, 'cdim': C_dim,
                     'scaler_min': scaler_min, 'scaler_max': scaler_max}

    X_train, Y_train = X[:int(src_data.shape[0] * train_ratio)], Y[:int(src_data.shape[0] * train_ratio)]
    X_test, Y_test = X[-int(src_data.shape[0] * test_ratio):], Y[-int(src_data.shape[0] * test_ratio):]
    return X_train, Y_train, X_test, Y_test, custom_config


def train_ddpm_msr():
    epochs = 200
    T = 20
    use_ema = False
    warmup_epoch = 5

    dataset_path = "../datasets/80c_20w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    M, W = custom_config['M'], custom_config['W']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=M, proj_dim=128, cond_dim=custom_config['sfn'] * M,# + custom_config['cdim'],
                   dims=(64, 32, 16, 8), is_attn=(False, False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, M, W, alphas, device, (1, M), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.apply(init_weights)
    diffusion_model.to(device)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150])

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


def custom_decoder(Y_pred):
    """
    Customized decoder for SMR.
    """
    Y_pred_decoded = (Y_pred - Y_pred.min()) / (Y_pred.max() - Y_pred.min())
    Y_pred_decoded = torch.softmax(Y_pred_decoded, dim=1)
    return Y_pred_decoded


@torch.no_grad()
def load_test_msr():
    T = 20
    omega = 150

    dataset_path = "../datasets/80c_20w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
    M, W = custom_config['M'], custom_config['W']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=M, proj_dim=128, cond_dim=custom_config['sfn'] * M,  # + custom_config['cdim'],
                   dims=(64, 32, 16, 8), is_attn=(False, False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, M, W, alphas, device, (1, M), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ready_models/ClassifierFree/80c_20w_20240109171558.pt"))
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
    Y_pred_decoded = W * custom_decoder(Y_pred)
    pred_rate = torch.sum(torch.log2(1.0 + Y_pred_decoded * X_test_tensor), dim=1)
    true_rate = torch.sum(torch.log2(1.0 + Y_test_tensor * X_test_tensor), dim=1)

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred_decoded[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_rate:\n", pred_rate[:20])
    print("true_rate:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))


@torch.no_grad()
def load_test_msr_debug():
    """
    Load ready model for debug especially.
    """
    T = 20
    omega = 150

    dataset_path = "../datasets/3c_10w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    M, W = custom_config['M'], custom_config['W']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=M, proj_dim=16, cond_dim=custom_config['sfn'] * M,# + custom_config['cdim'],
                   dims=(16, 8, 4), is_attn=(False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, M, W, alphas, device, (1, M), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ready_models/ClassifierFree/3c_10w_20240109165803.pt"))
    diffusion_model.to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    want2look = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Y_pred = diffusion_model.sample(X_test_tensor[want2look], omega)
    for i in range(Y_pred.shape[0]):
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(X_test_tensor[i], Y_pred[i])
        for j in range(diffusion_model.y_i_record.shape[0]):
            print(diffusion_model.y_i_record[j, i, :], diffusion_model.eps_i_record[j, i, :])

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


if __name__ == "__main__":
    print("########## Classifier-Free guidance diffusion for Computation Offloading. ##########")
    # diffusion_model = train_ddpm_msr()
    #
    # torch.save(diffusion_model.state_dict(), f"../ready_models/ClassifierFree/{diffusion_model.M}c_{int(diffusion_model.W)}w_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")
    #
    # load_test_msr()

    load_test_msr_debug()
