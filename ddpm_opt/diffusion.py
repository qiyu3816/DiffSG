"""
A customized implementation of Denoising Diffusion Probabilistic Models (DDPM) from the paper
Denoising Diffusion Probabilistic Models 2020
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from functools import partial

from ddpm_opt.ema import ExponentialMovingAverage

def generate_cosine_schedule(T, s=0.008):
    """
    按总步数和cos函数生成betas
    :param T:
    :param s:
    :return:
    """
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)
    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []
    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.84))
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    """
    按上下限和总步数生成betas
    :param T:
    :param low:
    :param high:
    :return:
    """
    return np.linspace(low, high, T)

@torch.no_grad()
def noise_single_sample(target_sum, size, device='cpu'):
    """
    Generate noise for a single sample with the shape (1, size), the sum of noise values is equal to target_num.
    Only used in process noise because it's negative-enabled.
    :param target_sum: Target sum of noise values.
    :param size: The noise size.
    """
    noise = np.random.dirichlet(np.ones(size) * 3, size=1) - 1 / size + target_sum / size
    noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
    return noise_tensor

@torch.no_grad()
def custom_noise_sample(target_sum, shape, device='cpu', enable_neg=True):
    """
    Customized noise random generation, all values can be positive or negative float numbers and the sum is target_sum.
    This function can be used for solution initialization.
    :param target_sum: Target sum of noise values.
    :param shape: The noise shape.
    :param enable_neg: If True, the noise value can be negative.
    """
    if enable_neg:
        noise = np.random.dirichlet(np.ones(shape[1]), size=1) - 1 / shape[1] + target_sum / shape[1]
    else:
        noise = np.random.dirichlet(np.ones(shape[1]), size=1) * target_sum
    noise = np.atleast_2d(noise)
    for i in range(shape[0] - 1):
        if enable_neg:
            tmp = np.random.dirichlet(np.ones(shape[1]), size=1) - 1 / shape[1] + target_sum / shape[1]
        else:
            tmp = np.random.dirichlet(np.ones(shape[1]), size=1) * target_sum
        noise = np.concatenate((noise, tmp), axis=0)
    noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
    return noise_tensor

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

@torch.no_grad()
def step_cost_calc(y_0, x_0, lambdas=torch.tensor([1.0, 0.05, 0.05, 1.0])):
    """
    :param y_0: the final resource allocation
    :param x_0: conditional features ((1, 3) per node), 0 common features
    :param lambdas: 4 lagrange multipliers for the loss function
    """
    # y = 1.0 / 2.0 * (y_0 - torch.mean(y_0)) / torch.std(y_0) + 0.5
    y = torch.softmax(y_0, dim=1)
    y = y + 0.00001
    x_0_inverse_scale = x_0 * (9.99927554792418 - 0.0015867173453851023) + 9.99927554792418  # new de-abnormal
    # x_0_inverse_scale = x_0 * (3228609869612.1245 - 0.0015867173453851023) + 0.0015867173453851023  # new
    # x_0_inverse_scale = x_0 * (31341.582234755024 - 2.6147616e-06) + 2.6147616e-06

    D_y_0 = torch.where(y > 0.1, 1, 0)

    local_cost = torch.index_select(x_0_inverse_scale, 1, torch.tensor([i for i in range(0, x_0_inverse_scale.shape[1], 3)], device=x_0_inverse_scale.device))
    offload_transition_cost = torch.index_select(x_0_inverse_scale, 1, torch.tensor([i for i in range(1, x_0_inverse_scale.shape[1], 3)], device=x_0_inverse_scale.device))
    ideal_offload_execution_cost = torch.index_select(x_0_inverse_scale, 1, torch.tensor([i for i in range(2, x_0_inverse_scale.shape[1], 3)], device=x_0_inverse_scale.device))

    # f(D,R)  #############
    total_cost_t_0 = torch.sum((1 - D_y_0) * local_cost +
                               D_y_0 * (offload_transition_cost + ideal_offload_execution_cost / y), dim=1)

    opt_loss = lambdas[0] * total_cost_t_0 #+ lambdas[3] * g5

    return opt_loss, y

@torch.no_grad()
def step_sum_rate(p_0, g_0):
    """
    :param p_0: final p, power allocation (batch_size, 3)
    :param g_0: current channel gains (batch_size, 4)
    """
    p_0 = p_0 * 10
    p_0_sum = torch.sum(p_0, dim=1)
    for i in range(p_0.shape[1]):
        p_0[:, i] = p_0[:, i] - p_0[:, i] / p_0_sum * (p_0_sum - 10.0)

    r_0 = torch.sum(torch.log2(1.0 + p_0 * g_0[:, :p_0.shape[1]]), dim=1)

    return r_0, p_0

class DiffusionOpt(nn.Module):
    """
    Conditional diffusion to make inference of the optimal y
    based on the given x and the target function f(x,y).
    """
    def __init__(self,
                 T,
                 model,
                 alphas,
                 task,
                 custom_config=None,
                 ema_decay=0.9999,
                 ema_start=1000,
                 ema_update_rate=5,
                 debug=False):
        super(DiffusionOpt, self).__init__()
        self.T = T
        self.model = model
        self.task = task
        self.custom_config = custom_config
        self.debug = debug

        betas = 1.0 - alphas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)
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

        self.loss_show_idx = 1

    def forward_process_y(self, y_0, t, noise):
        """
        The forward process to add noise on y_0.
        """
        return self.sqrt_alphas_cumprod[t].T * y_0 \
               + self.sqrt_one_minus_alphas_cumprod[t].T * noise

    @torch.no_grad()
    def custom_forward_process(self, y, t):
        """
        :param y: The sample being perturbed.
        :param t: The noising time step.
        """
        y_t = torch.zeros_like(y, device=self.device)
        for i in range(y.shape[0]):
            # cur_noise = noise_single_sample(0, y.shape[1], self.device)
            cur_noise = torch.zeros_like(y[i], device=self.device)
            if int(y[i][0]) == 1:
                cur_noise[0] = - random.random()
                cur_noise[1] = random.random()
                cur_noise[2] = - cur_noise[0] - cur_noise[1]
            elif int(y[i][1]) == 1:
                cur_noise[1] = - random.random()
                cur_noise[0] = random.random()
                cur_noise[2] = - cur_noise[0] - cur_noise[1]
            elif int(y[i][2]) == 1:
                cur_noise[2] = - random.random()
                cur_noise[0] = random.random()
                cur_noise[1] = - cur_noise[0] - cur_noise[2]
            cur_noise = torch.atleast_2d(cur_noise)

            cur_y = self.sqrt_alphas_cumprod[t[0][i]].T * y[i] + self.sqrt_one_minus_alphas_cumprod[t[0][i]].T * cur_noise
            # cur_y = torch.where(cur_y > 1, 1, cur_y)
            # cur_y = torch.where(cur_y < 0, 0, cur_y)
            # con = 1
            # while not (torch.where(cur_y < 1, 1, 0).all() and torch.where(cur_y > 0, 1, 0).all()):
            #     if con % 200 == 0:
            #         print(cur_noise, end=" ")
            #         print(cur_y, end=" ")
            #     cur_noise = noise_single_sample(0, y.shape[1], self.device)
            #     cur_y = self.sqrt_alphas_cumprod[t[0][i]].T * y[i] + self.sqrt_one_minus_alphas_cumprod[t[0][i]].T * cur_noise
            #     if con % 200 == 0:
            #         print(cur_y)
            #     con += 1
            y_t[i] = cur_y
            # if i % 1000 == 0:
            #     print(f"forward finish {i}")
            if i == 0:
                noise = cur_noise
            else:
                noise = torch.concatenate((noise, cur_noise), dim=0)
        return y_t, noise

    @torch.no_grad()
    def perturb_sample(self, y):
        """
        Return perturbed y and some related params.
        :param y: Input source label.
        """
        t = torch.randint(low=0, high=self.T, size=(1, y.shape[0]), device=self.device)
        y_t, noise = self.custom_forward_process(y, t)
        return y_t, t, noise

    def train(self, y, x, random_time=300):
        """
        Train function to return the loss.
        :param y: The given ground truth.
        :param x: The given condition vector.
        :param random_time: The num of perturbed samples generated on one sample.
        """
        self.device = x.device
        if self.task == "CONV_CO":
            self.special_feature_num, self.common_feature_num = self.custom_config['sfn'], self.custom_config['cfn']
            self.node_num = self.task_tt_expand = (x.shape[1] - self.common_feature_num) // self.special_feature_num
        elif self.task == "MAX SUM RATE":
            self.channel_num = self.task_tt_expand = x.shape[1]

        y_t, t, noise = self.perturb_sample(y)
        # for i in tqdm(range(random_time)):
        #     if i == 0:
        #         y_t, t, noise = self.perturb_sample(y)
        #     else:
        #         cur_y_t, cur_t, cur_noise = self.perturb_sample(y)
        #         y_t = torch.cat((y_t, cur_y_t), dim=0)
        #         t = torch.concatenate((t, cur_t), dim=1)
        #         noise = torch.cat((noise, cur_noise), dim=0)

        if self.task == "CONV_CO":
            if self.custom_config is not None and 'scaler' in self.custom_config.keys():
                scaler = self.custom_config['scaler']
                x_scaled = torch.tensor(scaler.raw2scaled(x[:, :-7]), dtype=torch.float32)
                x_scaled = torch.concatenate((x_scaled, x[:, -7:]), dim=1)
                x_scaled[:, -7] = scaler.lower + (scaler.upper - scaler.lower) * (x[:, -7] - scaler.mins[2]) \
                                  / (scaler.maxs[2] - scaler.mins[2])
            else:
                x_scaled = x
        elif self.task == "MAX SUM RATE":
            x_scaled = x

        # x_scaled = x_scaled.repeat(random_time, 1)

        estimated_noise = self.model(y_t, t, x_scaled)
        loss = F.mse_loss(estimated_noise, noise)
        # print(y[1], y_t[1], noise[1], x_scaled[1], t[0][1], estimated_noise[1])
        # loss = F.mse_loss((y_t - self.betas[t].repeat(y_t.shape[1], 1).T / self.sqrt_one_minus_alphas_cumprod[t].repeat(y_t.shape[1], 1).T * estimated_noise) * self.reciprocal_sqrt_alphas[t].repeat(y_t.shape[1], 1).T, y)
        return loss

    @torch.no_grad()
    def custom_guidance(self, x_scaled):
        local_cost = torch.index_select(x_scaled, 1, torch.tensor([i for i in range(0, x_scaled.shape[1], 3)], device=x_scaled.device))
        max_indices = []
        for i in range(local_cost.shape[0]):
            max_indices.append(torch.argmax(local_cost[i]))
        for i in range(local_cost.shape[0]):
            local_cost[i][max_indices[i]] = random.random()
            for j in range(local_cost.shape[1]):
                if not j == max_indices[i]:
                    local_cost[i][j] = - local_cost[i][max_indices[i]] / 2

        offload_transition_cost = torch.index_select(x_scaled, 1, torch.tensor([i for i in range(1, x_scaled.shape[1], 3)], device=x_scaled.device))
        ideal_offload_execution_cost = torch.index_select(x_scaled, 1, torch.tensor([i for i in range(2, x_scaled.shape[1], 3)], device=x_scaled.device))
        offload_transition_cost = offload_transition_cost + ideal_offload_execution_cost
        max_indices = []
        for i in range(offload_transition_cost.shape[0]):
            max_indices.append(torch.argmax(offload_transition_cost[i]))
        for i in range(offload_transition_cost.shape[0]):
            offload_transition_cost[i][max_indices[i]] = - random.random()
            for j in range(offload_transition_cost.shape[1]):
                if not j == max_indices[i]:
                    offload_transition_cost[i][j] = - offload_transition_cost[i][max_indices[i]] / 2
        return 6 * offload_transition_cost + 3 * local_cost

    @torch.no_grad()
    def custom_denoise(self, y_t, estimated_noise, step, x_scaled):
        """
        Customized denoise to limit the de-noised sample into valid solution space.
        """
        # guidance = - self.custom_guidance(x_scaled)
        for i in range(y_t.shape[0]):
            cur_noise = noise_single_sample(0, y_t.shape[1], self.device)
            if i == 0:
                noise = cur_noise
            else:
                noise = torch.concatenate((noise, cur_noise), dim=0)
        if step == 0:
            noise = torch.zeros_like(y_t, device=self.device)
        y_t = (y_t - 4 * self.betas[step] / self.sqrt_one_minus_alphas_cumprod[step] * estimated_noise) * self.reciprocal_sqrt_alphas[step] \
              + (1.0 - self.alphas_cumprod[step - 1 if step - 1 >= 0 else 0]) / (1.0 - self.alphas_cumprod[step]) * noise
        # print("noise=", noise[1], end="")
        if self.task == "MAX SUM RATE":
            y_t = torch.where(y_t > 1, 1, y_t)
            y_t = torch.where(y_t < 0, 0.00001, y_t)
        return y_t

    def forward(self, x, y=None):
        """
        Input the raw x and y for train.
        :param x: the given conditional parameters (batch_size, in_dim=special_dim+common_dim)
        :param y: the real label (batch_size, out_dim)
        """
        self.device = x.device

        t = torch.full(size=(1, x.shape[0]), fill_value=self.T - 1, device=self.device)
        if y is not None:
            noise = torch.randn_like(y)
            y_t = self.forward_process_y(y, t, noise)

        if self.task == "CONV_CO":
            self.special_feature_num, self.common_feature_num = self.custom_config['sfn'], self.custom_config['cfn']
            y_t = custom_noise_sample(1, (x.shape[0], (x.shape[1] - self.common_feature_num) // self.special_feature_num), device=self.device, enable_neg=False)
            if self.custom_config is not None and 'scaler' in self.custom_config.keys():
                scaler = self.custom_config['scaler']
                x_scaled = torch.tensor(scaler.raw2scaled(x[:, :-7]), dtype=torch.float32)
                x_scaled = torch.concatenate((x_scaled, x[:, -7:]), dim=1)
                x_scaled[:, -7] = scaler.lower + (scaler.upper - scaler.lower) * (x[:, -7] - scaler.mins[2]) \
                                  / (scaler.maxs[2] - scaler.mins[2])
            else:
                x_scaled = x
        elif self.task == "MAX SUM RATE":
            y_t = custom_noise_sample(1, (x.shape[0], x.shape[1]), device=self.device, enable_neg=False)
            x_scaled = x

        if self.debug:
            self.loss_record = []
            if self.task == "CONV_CO":
                cur_loss, cur_y = step_cost_calc(y_t, x_scaled)
            elif self.task == "MAX SUM RATE":
                cur_loss, cur_y = step_sum_rate(y_t, x_scaled)
            self.loss_record.append({'loss': cur_loss[self.loss_show_idx].item(), 'y': cur_y[self.loss_show_idx]})
            print(f"time step {self.T}, loss={cur_loss[self.loss_show_idx].item()}, y={cur_y[self.loss_show_idx]}")
        self.denoise_process_record = []
        for i in range(self.T - 1, -1, -1):
            estimated_noise = self.model(y_t, torch.full(size=(1, x.shape[0]), fill_value=i, device=self.device), x_scaled)
            # print(y_t[1], estimated_noise[1], x_scaled[1], end=" ")
            # print(estimated_noise[:10])
            # print(x_scaled[:10] * (3228609869612.1245 - 0.0015867173453851023) + 0.0015867173453851023)
            # y_t = (y_t - self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i] * estimated_noise) * \
            #       self.reciprocal_sqrt_alphas[i] \
            #       + (1.0 - self.alphas_cumprod[i - 1 if i - 1 >= 0 else 0]) / (1.0 - self.alphas_cumprod[i]) * torch.randn_like(y_t)
            y_t = self.custom_denoise(y_t, estimated_noise, i, x_scaled)

            min_value = y_t.min()
            max_value = y_t.max()
            y_t = (y_t - min_value) / (max_value - min_value)

            # print("after=", y_t[1])
            if self.debug:
                if self.task == "CONV_CO":
                    cur_loss, cur_y = step_cost_calc(y_t, x_scaled)
                elif self.task == "MAX SUM RATE":
                    cur_loss, cur_y = step_sum_rate(y_t, x_scaled)
                self.loss_record.append({'loss': cur_loss[self.loss_show_idx].item(), 'y': cur_y[self.loss_show_idx]})
                torch.set_printoptions(precision=4, sci_mode=False)
                self.denoise_process_record.append(cur_y[self.loss_show_idx].cpu().numpy())
                print(f"time step {i}, loss={cur_loss[self.loss_show_idx].item()}, y={cur_y[self.loss_show_idx]}")
        return y_t