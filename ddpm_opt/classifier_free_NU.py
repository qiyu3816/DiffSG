"""
The Classifier-Free Guidance version of DIFFSG specified for NOMA-UAV (NU).
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


width = 400
height = 400

def condition_C(y, x, width, height, P_sum):
    """
    Calculate the customized objectives and constraints in numerical conditions.
    :param y: (batch_size, channel_num)
    :param x: (batch_size, sfn * node_num)
    :return: (batch_size, sfn * node_num + c_dim)
    """
    Y_pred_decoded = torch.zeros_like(y, device=y.device)
    Y_pred_decoded[:, :2] = (y[:, :2] - torch.min(y[:, :2])) / (torch.max(y[:, :2]) - torch.min(y[:, :2]))
    Y_pred_decoded[:, 0] *= width
    Y_pred_decoded[:, 1] *= height
    Y_pred_decoded[:, 2:] = torch.softmax(Y_pred_decoded[:, 2:], dim=1) * P_sum

    K = Y_pred_decoded.shape[1] - 2
    X = torch.zeros_like(x)
    for i in range(K):
        X[:, 2 * i] *= width
        X[:, 2 * i + 1] *= height

    sigma_sq = 110
    rou_0 = 60
    H = 150
    K = Y_pred_decoded.shape[1] - 2

    h = torch.zeros_like(Y_pred_decoded[:, 2:], device=Y_pred_decoded.device)
    sinr = torch.zeros_like(Y_pred_decoded[:, 2:], device=Y_pred_decoded.device)
    for i in range(Y_pred_decoded.shape[0]):
        for j in range(K):
            h[i, j] = torch.sqrt(rou_0 / (H ** 2 + (X[i, j * 2] - Y_pred_decoded[i, 0]) ** 2 + (
                        X[i, j * 2 + 1] - Y_pred_decoded[i, 1]) ** 2))
        sorted_indices = torch.argsort(-h[i])
        for index, jj in enumerate(sorted_indices):
            if index == 0:
                sinr[i, jj] = Y_pred_decoded[i, 2 + jj] * (h[i, jj] ** 2) / sigma_sq
            else:
                sinr[i, jj] = Y_pred_decoded[i, 2 + jj] / (
                            torch.sum(Y_pred_decoded[i, 2 + sorted_indices[:index]]) + sigma_sq / (h[i, jj] ** 2))
    rates = torch.sum(torch.log2(1 + sinr), dim=1)[:, None]

    x = torch.cat((x, rates), dim=1)
    return x


# DDPM
class DDPM(nn.Module):
    """
    DDPM in the version of "Classifier-Free Diffusion Guidance".
    """

    def __init__(self,
                 T,
                 model,
                 K,
                 P_sum,
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
        self.K = K
        self.P_sum = P_sum
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

        self.record_denoise_path = False

    def forward(self, y, cond):
        ts = torch.randint(low=0, high=self.T, size=(1, y.shape[0]), device=self.device)
        noise = torch.randn_like(y, device=self.device)
        y_t = self.sqrt_alphas_cumprod[ts, None] * y + self.sqrt_one_minus_alphas_cumprod[ts, None] * noise
        y_t = torch.squeeze(y_t)
        # cond = condition_C(y_t, cond, self.custom_config['width'], self.custom_config['height'], self.P_sum)

        cond_mask = torch.bernoulli(torch.fill(torch.zeros(cond.shape[0], device=self.device), 1 - self.uncond_prob))[:, None]

        estimated_noise = self.model(y_t, ts / self.T, cond, cond_mask)
        if random.random() < 0.005:
            print(f"noise {noise[1, :]}, estimated_noise {estimated_noise[1, :]}")
        return F.mse_loss(noise, estimated_noise)

    def sample(self, cond, omega=1.0):
        y_t = torch.randn(cond.shape[0], *self.data_size).to(self.device)
        y_t = torch.squeeze(y_t)

        # cond = condition_C(y_t, cond, self.custom_config['width'], self.custom_config['height'], self.P_sum)
        cond_mask_0 = torch.zeros(cond.shape[0], device=self.device)[:, None]
        cond_mask_1 = torch.ones(cond.shape[0], device=self.device)[:, None]

        y_i_record = []  # keep track of generated steps in case want to plot something
        eps_i_record = []
        for i in range(self.T - 1, -1, -1):
            # cond = condition_C(y_t, cond[:, :-1], self.custom_config['width'], self.custom_config['height'], self.P_sum)
            eps_0 = self.model(y_t, torch.full(size=(1, cond.shape[0]), fill_value=i, device=self.device) / self.T, cond, cond_mask_0)
            eps_1 = self.model(y_t, torch.full(size=(1, cond.shape[0]), fill_value=i, device=self.device) / self.T, cond, cond_mask_1)

            noise = torch.randn(cond.shape[0], *self.data_size).to(self.device) if i > 1 else 0
            noise = torch.squeeze(noise) if i > 1 else 0

            eps = (1 + omega) * eps_1 - omega * eps_0
            y_t = (y_t - self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i] * eps) * self.reciprocal_sqrt_alphas[i] \
                  + (1.0 - self.alphas_cumprod[i - 1 if i - 1 >= 0 else 0]) / (1.0 - self.alphas_cumprod[i]) * noise

            if i > self.T - 5:  # Normalization in case that solution value explosion in the early sampling.
                y_t = (y_t - torch.mean(y_t)) / torch.sqrt(torch.var(y_t))

            if self.record_denoise_path:
                y_i_record.append(y_t.detach().cpu().numpy())
                eps_i_record.append(eps.detach().cpu().numpy())

        if self.record_denoise_path:
            self.y_i_record = np.array(y_i_record)
            for j in range(self.y_i_record.shape[0]):
                self.y_i_record[j] = custom_decoder(
                    torch.tensor(self.y_i_record[j], dtype=torch.float32), width, height, self.P_sum).detach().cpu().numpy()
            self.y_i_record = self.y_i_record.transpose(1, 0, 2).reshape(self.y_i_record.shape[1], -1)
            self.eps_i_record = np.array(eps_i_record)
            self.eps_i_record = self.eps_i_record.transpose(1, 0, 2).reshape(self.eps_i_record.shape[1], -1)
        return y_t


# Dataset and preprocessing
def nu_data_load(dataset_path, width, height):
    """
    Load, scale and split NOMA-UAV dataset.
    """
    shuffle = False
    train_ratio, test_ratio = 0.7, 0.3
    C_dim = 1  # only objective function value

    src_csv = pd.read_csv(dataset_path, header=None)
    src_data = np.array(src_csv)
    if shuffle:
        np.random.shuffle(src_data)
    K = (src_data.shape[1] - 3) // 3
    P_sum = float(dataset_path.split('_')[-2][:-2])
    X, Y, R = src_data[:, :2 * K], src_data[:, 2 * K:2 + 3 * K], src_data[:, -1]

    for i in range(K):
        X[:, 2 * i] = X[:, 2 * i] / width
        X[:, 2 * i + 1] = X[:, 2 * i + 1] / height
        Y[:, 2 + i] = Y[:, 2 + i] / P_sum
    Y[:, 0] = Y[:, 0] / width
    Y[:, 1] = Y[:, 1] / height
    custom_config = {'K': K, 'P_sum': P_sum, 'cdim': C_dim, 'width': width, 'height': height}

    X_train, Y_train = X[:int(src_data.shape[0] * train_ratio)], Y[:int(src_data.shape[0] * train_ratio)]
    X_test, Y_test, R_test = X[-int(src_data.shape[0] * test_ratio):], Y[-int(src_data.shape[0] * test_ratio):], R[-int(src_data.shape[0] * test_ratio):]
    return X_train, Y_train, X_test, Y_test, R_test, custom_config


def train_ddpm_nu():
    epochs = 200
    T = 20
    use_ema = False
    warmup_epoch = 5

    width = 400
    height = 400
    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    K, P_sum = custom_config['K'], custom_config['P_sum']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=2 + K, proj_dim=32, cond_dim=2 * K,# + custom_config['cdim'],
                   dims=(32, 16, 8), is_attn=(False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, K, P_sum, alphas, device, (1, 2 + K), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.apply(init_weights)
    diffusion_model.to(device)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.004)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 200])

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


def custom_decoder(Y_pred, width, height, P_sum):
    """
    Customized decoder for NOMA-UAV.
    """
    Y_pred_decoded = torch.zeros_like(Y_pred, device=Y_pred.device)
    Y_pred_decoded[:, :2] = (Y_pred[:, :2] - torch.min(Y_pred[:, :2])) / (torch.max(Y_pred[:, :2]) - torch.min(Y_pred[:, :2]))
    Y_pred_decoded[:, 0] *= width
    Y_pred_decoded[:, 1] *= height
    Y_pred_decoded[:, 2:] = torch.softmax(Y_pred[:, 2:], dim=1) * P_sum
    return Y_pred_decoded


def rate_calc(Y_pred_decoded, X):
    """
    Calculate the final rates with given X and predicted Y.
    :param Y_pred_decoded: (sample_num, 2 + K)
    :param X: (sample_num, 2 * K)
    :return: rates(sample_num)
    """
    sigma_sq = 110
    rou_0 = 60
    H = 150
    K = Y_pred_decoded.shape[1] - 2

    h = torch.zeros_like(Y_pred_decoded[:, 2:], device=Y_pred_decoded.device)
    sinr = torch.zeros_like(Y_pred_decoded[:, 2:], device=Y_pred_decoded.device)
    for i in range(Y_pred_decoded.shape[0]):
        for j in range(K):
            h[i, j] = torch.sqrt(rou_0 / (H ** 2 + (X[i, j * 2] - Y_pred_decoded[i, 0]) ** 2 + (X[i, j * 2 + 1] - Y_pred_decoded[i, 1]) ** 2))
        sorted_indices = torch.argsort(-h[i])
        for index, jj in enumerate(sorted_indices):
            if index == 0:
                sinr[i, jj] = Y_pred_decoded[i, 2 + jj] * (h[i, jj] ** 2) / sigma_sq
            else:
                sinr[i, jj] = Y_pred_decoded[i, 2 + jj] / (torch.sum(Y_pred_decoded[i, 2 + sorted_indices[:index]]) + sigma_sq / (h[i, jj] ** 2))
    rates = torch.sum(torch.log2(1 + sinr), dim=1)
    return rates


@torch.no_grad()
def load_test_nu(ckpt_path):
    T = 20
    omega = 500

    width = 400
    height = 400
    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
    K, P_sum = custom_config['K'], custom_config['P_sum']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=2 + K, proj_dim=32, cond_dim=2 * K,  # + custom_config['cdim'],
                   dims=(32, 16, 8), is_attn=(False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, K, P_sum, alphas, device, (1, 2 + K), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load(ckpt_path))
    diffusion_model.to(device)

    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = diffusion_model.sample(x, omega)
        else:
            Y_pred = torch.cat((Y_pred, diffusion_model.sample(x, omega)))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    for i in range(K):
        X_test_tensor[:, 2 * i] *= width
        X_test_tensor[:, 2 * i + 1] *= height
    Y_pred_decoded = custom_decoder(Y_pred, width, height, P_sum)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)
    Y_test_tensor[:, 0] *= width
    Y_test_tensor[:, 1] *= height
    Y_test_tensor[:, 2:] *= P_sum
    pred_rate = rate_calc(Y_pred_decoded, X_test_tensor)
    true_rate = rate_calc(Y_test_tensor, X_test_tensor)

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)
    print("Y_pred:\n", Y_pred_decoded[:20])
    print("Y_test:\n", Y_test_tensor[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_rate:\n", pred_rate[:20])
    print("true_rate:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))


@torch.no_grad()
def load_test_nu_debug():
    """
    Load ready model for debug especially.
    """
    T = 20
    omega = 500

    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    K, P_sum = custom_config['K'], custom_config['P_sum']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=2 + K, proj_dim=32, cond_dim=2 * K,  # + custom_config['cdim'],
                   dims=(32, 16, 8), is_attn=(False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, K, P_sum, alphas, device, (1, 2 + K), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ckpts/ddpm_nu_3u.pt"))
    diffusion_model.to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    diffusion_model.record_denoise_path = True
    _ = diffusion_model.sample(X_test_tensor, omega)
    df = pd.DataFrame(diffusion_model.y_i_record)
    df.to_csv(f"../results/nu_denoise_path.csv", header=None, index=False)
    print(f"Trajectory generating finished, {diffusion_model.y_i_record.shape[0]} samples stored.")


if __name__ == "__main__":
    print("########## Classifier-Free guidance diffusion for NOMA-UAV. ##########")
    # diffusion_model = train_ddpm_nu()
    #
    # ckpt_path = f"../ckpts/ddpm_nu_{diffusion_model.K}u_{datetime.datetime.now():%Y%m%d%H%M%S}.pt"
    # torch.save(diffusion_model.state_dict(), ckpt_path)

    # ckpt_path = "../ckpts/ddpm_nu_3u.pt"
    # load_test_nu(ckpt_path)
    #
    load_test_nu_debug()
