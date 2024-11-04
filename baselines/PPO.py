"""
Baseline Proximal Policy Optimization (PPO) tested on CO, MSR and NU.
"""
import random
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.distributions import normal, Normal

from ddpm_opt.classifier_free_CO import co_data_load, cost_calc
from ddpm_opt.classifier_free_MSR import msr_data_load
from ddpm_opt.classifier_free_NU import nu_data_load, rate_calc, custom_decoder

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: x.shape[0]
        :param action_dim: y.shape[0]
        """
        super(PPOAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # critic to estimate the state value function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        # actor to predict the action
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp()
        dist = normal.Normal(mu, std)
        return value, dist


def calc_advantage(rewards, values, gamma=0.99):
    returns = []
    for r in rewards:
        discounted_sum = r + (gamma * 3.8)
        returns.append(discounted_sum)
    returns = torch.tensor(returns, device=values.device, dtype=torch.float32).unsqueeze(dim=1)
    advantages = returns - values
    return advantages, returns

def clipped_surrogate_objective_loss(ratio, advantage, epsilon=0.2):
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    loss = -torch.min(unclipped, clipped).mean()
    return loss


@torch.no_grad()
def co_env_step(cur_states, actions, ground_truth, custom_config):
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']

    next_state = cur_states.detach().clone()
    costs = cost_calc(cur_states * (scaler_max - scaler_min) + scaler_min, actions)

    gt_costs = cost_calc(cur_states * (scaler_max - scaler_min) + scaler_min, ground_truth)
    diff = torch.abs(costs - gt_costs) + 0.1
    rewards = 1 / diff

    return next_state, rewards

def ppo_co():
    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    node_num = Y_train.shape[1]

    state_dim = node_num * 3
    action_dim = node_num
    epochs = 200

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim, action_dim)
    agent.to(device)
    # agent.load_state_dict(torch.load("../ready_models/PPO/3n_co_20240117171154.pt"))

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=0.005)
    actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(actor_optimizer, [20, 100])
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=0.005)
    critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, [20, 100])

    X_train_tensor = torch.tensor(X_train, device=device, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, device=device, dtype=torch.float32)

    with torch.no_grad():
        means, stds = 0.5 * torch.ones_like(Y_train_tensor, device=device, dtype=torch.float32), 0.2 * torch.ones_like(Y_train_tensor, device=device, dtype=torch.float32)
        norm_dist = Normal(means, stds)
        actions = norm_dist.sample()
        old_prob = norm_dist.log_prob(actions)
        dataset = data.TensorDataset(X_train_tensor, Y_train_tensor, old_prob)
        data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)

    for it in range(epochs):
        X_train_next, Y_train_next, old_log_prob_ne = None, None, None
        epoch_a_loss = 0
        epoch_c_loss = 0
        epoch_reward = 0
        epoch_sample_num = 0
        for x, y, old_log_prob in tqdm(data_loader):
            values, distributions = agent(x)
            actions = distributions.sample()
            new_log_prob = distributions.log_prob(actions)
            actions = torch.softmax(actions, dim=1)

            next_state, rewards = co_env_step(x, actions, y, custom_config)

            advantages, returns = calc_advantage(rewards, values)
            ratio = (new_log_prob - old_log_prob).exp()

            actor_loss = clipped_surrogate_objective_loss(ratio, advantages)
            actor_loss.backward(retain_graph=True)
            critic_loss = F.mse_loss(values, returns)
            critic_loss.backward()

            actor_optimizer.step()
            actor_optimizer.zero_grad()
            critic_optimizer.step()
            critic_optimizer.zero_grad()

            epoch_a_loss += actor_loss.item()
            epoch_c_loss += critic_loss.item()
            epoch_reward += torch.sum(rewards)
            epoch_sample_num += x.shape[0]

            with torch.no_grad():
                if X_train_next is None:
                    X_train_next, Y_train_next, old_log_prob_ne = x, y, new_log_prob
                else:
                    X_train_next = torch.concatenate((X_train_next, x))
                    Y_train_next = torch.concatenate((Y_train_next, y))
                    old_log_prob_ne = torch.concatenate((old_log_prob_ne, new_log_prob))
        with torch.no_grad():
            dataset = data.TensorDataset(X_train_next, Y_train_next, old_log_prob_ne)
            data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
        actor_lr_scheduler.step()
        critic_lr_scheduler.step()

        print(f"Epoch: {it}, Actor loss: {epoch_a_loss / epoch_sample_num}, Critic loss: {epoch_c_loss / epoch_sample_num}.")
        print(f"Reward: {epoch_reward / epoch_sample_num}")

    X_test_tensor = torch.tensor(X_test, device=device, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)
    dataset = data.TensorDataset(X_test_tensor, Y_test_tensor)
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            _, distributions = agent(x)
            action = distributions.sample()
            Y_pred = torch.softmax(action, dim=1)
        else:
            _, distributions = agent(x)
            action = distributions.sample()
            Y_pred = torch.cat((Y_pred, torch.softmax(action, dim=1)))

    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test_tensor = X_test_tensor * (scaler_max - scaler_min) + scaler_min
    pred_cost = cost_calc(X_test_tensor, Y_pred)
    true_cost = cost_calc(X_test_tensor, Y_test_tensor)

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_cost:\n", pred_cost[:20])
    print("true_cost:\n", true_cost[:20])
    print(f"exceeded ratio: {torch.sum(pred_cost) / torch.sum(true_cost)}")
    print("avg cost diff:\n", torch.mean(pred_cost - true_cost))

    torch.save(agent.state_dict(), f"../ready_models/PPO/{node_num}n_co_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")


@torch.no_grad()
def msr_env_step(cur_states, actions, ground_truth, custom_config):
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    W = custom_config['W']

    next_state = cur_states.detach().clone()
    rates = torch.sum(torch.log2(1.0 + actions * W * (cur_states * (scaler_max - scaler_min) + scaler_min)), dim=1)

    gt_rates = torch.sum(torch.log2(1.0 + ground_truth * W * (cur_states * (scaler_max - scaler_min) + scaler_min)), dim=1)
    diff = torch.abs(rates - gt_rates) + 0.01
    rewards = 1 / diff

    return next_state, rewards

def ppo_msr():
    dataset_path = "../datasets/3c_10w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    M, W = custom_config['M'], custom_config['W']
    Y_train /= W

    state_dim = M
    action_dim = M
    epochs = 100

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim, action_dim)
    agent.to(device)
    # agent.load_state_dict(torch.load("../ready_models/PPO/3c_msr_20240117180802.pt"))

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=0.005)
    actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(actor_optimizer, [20])
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=0.005)
    critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, [20])

    X_train_tensor = torch.tensor(X_train, device=device, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, device=device, dtype=torch.float32)

    with torch.no_grad():
        means, stds = 0.5 * torch.ones_like(Y_train_tensor, device=device, dtype=torch.float32), 0.2 * torch.ones_like(
            Y_train_tensor, device=device, dtype=torch.float32)
        norm_dist = Normal(means, stds)
        actions = norm_dist.sample()
        old_prob = norm_dist.log_prob(actions)
        dataset = data.TensorDataset(X_train_tensor, Y_train_tensor, old_prob)
        data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)

    for it in range(epochs):
        X_train_next, Y_train_next, old_log_prob_ne = None, None, None
        epoch_a_loss = 0
        epoch_c_loss = 0
        epoch_reward = 0
        epoch_sample_num = 0
        for x, y, old_log_prob in tqdm(data_loader):
            values, distributions = agent(x)
            actions = distributions.sample()
            new_log_prob = distributions.log_prob(actions)
            actions = torch.softmax(actions, dim=1)

            next_state, rewards = msr_env_step(x, actions, y, custom_config)

            advantages, returns = calc_advantage(rewards, values)
            ratio = (new_log_prob - old_log_prob).exp()

            actor_loss = clipped_surrogate_objective_loss(ratio, advantages)
            actor_loss.backward(retain_graph=True)
            critic_loss = F.mse_loss(values, returns)
            critic_loss.backward()

            actor_optimizer.step()
            actor_optimizer.zero_grad()
            critic_optimizer.step()
            critic_optimizer.zero_grad()

            epoch_a_loss += actor_loss.item()
            epoch_c_loss += critic_loss.item()
            epoch_reward += torch.sum(rewards)
            epoch_sample_num += x.shape[0]

            with torch.no_grad():
                if X_train_next is None:
                    X_train_next, Y_train_next, old_log_prob_ne = x, y, new_log_prob
                else:
                    X_train_next = torch.concatenate((X_train_next, x))
                    Y_train_next = torch.concatenate((Y_train_next, y))
                    old_log_prob_ne = torch.concatenate((old_log_prob_ne, new_log_prob))
        with torch.no_grad():
            dataset = data.TensorDataset(X_train_next, Y_train_next, old_log_prob_ne)
            data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
        actor_lr_scheduler.step()
        critic_lr_scheduler.step()

        print(f"Epoch: {it}, Actor loss: {epoch_a_loss / epoch_sample_num}, Critic loss: {epoch_c_loss / epoch_sample_num}.")
        print(f"Reward: {epoch_reward / epoch_sample_num}")

    X_test_tensor = torch.tensor(X_test, device=device, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)
    dataset = data.TensorDataset(X_test_tensor, Y_test_tensor)
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            _, distributions = agent(x)
            action = distributions.mean
            Y_pred = torch.softmax(action, dim=1)
        else:
            _, distributions = agent(x)
            action = distributions.mean
            Y_pred = torch.cat((Y_pred, torch.softmax(action, dim=1)))

    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test_tensor = X_test_tensor * (scaler_max - scaler_min) + scaler_min
    Y_pred *= W
    pred_rate = torch.sum(torch.log2(1.0 + Y_pred * X_test_tensor), dim=1)
    true_rate = torch.sum(torch.log2(1.0 + Y_test_tensor * X_test_tensor), dim=1)

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_rate:\n", pred_rate[:20])
    print("true_rate:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))

    torch.save(agent.state_dict(), f"../ready_models/PPO/{M}c_msr_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")


@torch.no_grad()
def nu_env_step(cur_states, actions, ground_truth, custom_config):
    width, height, P_sum = custom_config['width'], custom_config['height'], custom_config['P_sum']

    next_state = cur_states.detach().clone()
    real_states = torch.zeros_like(cur_states, device=actions.device, dtype=torch.float32)
    real_states[:, [0, 2, 4]] *= width
    real_states[:, [1, 3, 5]] *= height
    rates = rate_calc(actions, real_states)

    gt_rates = rate_calc(ground_truth, real_states)
    diff = torch.abs(rates - gt_rates) + 0.1
    rewards = 1 / diff

    return next_state, rewards

def ppo_nu():
    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    width, height = 400, 400
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    Y_train[:, 0] *= width
    Y_train[:, 1] *= height
    K, P_sum = custom_config['K'], custom_config['P_sum']
    Y_train[:, -3:] *= P_sum

    state_dim = K * 2
    action_dim = K + 2
    epochs = 50

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim, action_dim)
    agent.to(device)
    # agent.load_state_dict(torch.load("../ready_models/PPO/3u_nu_20240117204222.pt"))

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=0.005)
    actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(actor_optimizer, [20])
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=0.005)
    critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, [20])

    X_train_tensor = torch.tensor(X_train, device=device, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, device=device, dtype=torch.float32)

    with torch.no_grad():
        means, stds = 0.5 * torch.ones_like(Y_train_tensor, device=device, dtype=torch.float32), 0.2 * torch.ones_like(
            Y_train_tensor, device=device, dtype=torch.float32)
        norm_dist = Normal(means, stds)
        actions = norm_dist.sample()
        old_prob = norm_dist.log_prob(actions)
        dataset = data.TensorDataset(X_train_tensor, Y_train_tensor, old_prob)
        data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)

    for it in range(epochs):
        X_train_next, Y_train_next, old_log_prob_ne = None, None, None
        epoch_a_loss = 0
        epoch_c_loss = 0
        epoch_reward = 0
        epoch_sample_num = 0
        for x, y, old_log_prob in tqdm(data_loader):
            values, distributions = agent(x)
            actions = distributions.sample()
            new_log_prob = distributions.log_prob(actions)
            actions = torch.softmax(actions, dim=1)

            next_state, rewards = nu_env_step(x, custom_decoder(actions, width, height, P_sum), y, custom_config)

            advantages, returns = calc_advantage(rewards, values)
            ratio = (new_log_prob - old_log_prob).exp()

            actor_loss = clipped_surrogate_objective_loss(ratio, advantages)
            actor_loss.backward(retain_graph=True)
            critic_loss = F.mse_loss(values, returns)
            critic_loss.backward()

            actor_optimizer.step()
            actor_optimizer.zero_grad()
            critic_optimizer.step()
            critic_optimizer.zero_grad()

            epoch_a_loss += actor_loss.item()
            epoch_c_loss += critic_loss.item()
            epoch_reward += torch.sum(rewards)
            epoch_sample_num += x.shape[0]

            with torch.no_grad():
                if X_train_next is None:
                    X_train_next, Y_train_next, old_log_prob_ne = x, y, new_log_prob
                else:
                    X_train_next = torch.concatenate((X_train_next, x))
                    Y_train_next = torch.concatenate((Y_train_next, y))
                    old_log_prob_ne = torch.concatenate((old_log_prob_ne, new_log_prob))
        with torch.no_grad():
            dataset = data.TensorDataset(X_train_next, Y_train_next, old_log_prob_ne)
            data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
        actor_lr_scheduler.step()
        critic_lr_scheduler.step()

        print(f"Epoch: {it}, Actor loss: {epoch_a_loss / epoch_sample_num}, Critic loss: {epoch_c_loss / epoch_sample_num}.")
        print(f"Reward: {epoch_reward / epoch_sample_num}")

    X_test_tensor = torch.tensor(X_test, device=device, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)
    dataset = data.TensorDataset(X_test_tensor, Y_test_tensor)
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            _, distributions = agent(x)
            action = distributions.mean
            Y_pred = custom_decoder(action, width, height, P_sum)
        else:
            _, distributions = agent(x)
            action = distributions.mean
            Y_pred = torch.cat((Y_pred, custom_decoder(action, width, height, P_sum)))

    for i in range(K):
        X_test_tensor[:, 2 * i] *= width
        X_test_tensor[:, 2 * i + 1] *= height
    Y_test_tensor[:, 0] *= width
    Y_test_tensor[:, 1] *= height
    Y_test_tensor[:, 2:] *= P_sum

    pred_rate = rate_calc(Y_pred, X_test_tensor)
    true_rate = rate_calc(Y_test_tensor, X_test_tensor)

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test_tensor[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_rate:\n", pred_rate[:20])
    print("true_rate:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))

    torch.save(agent.state_dict(), f"../ready_models/PPO/{K}u_nu_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")


if __name__ == "__main__":

    # ppo_co()

    # ppo_msr()

    ppo_nu()
