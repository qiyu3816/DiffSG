"""
Baseline MTFNN implementation, which is tested on CO, MSR and NU.
"""
import random
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import datetime
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from ddpm_opt.classifier_free_CO import co_data_load, cost_calc
from ddpm_opt.classifier_free_MSR import msr_data_load
from ddpm_opt.classifier_free_NU import nu_data_load, rate_calc
from ddpm_opt.diffusion import init_weights


def mtfnn_co():
    dataset_path = "../datasets/3nodes_50000samples_new.csv"
    X_train, Y_train, X_test, Y_test, custom_config = co_data_load(dataset_path)
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    node_num = Y_train.shape[1]

    in_dim = node_num * 3
    out_dim = node_num
    epochs = 50

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(OrderedDict([
        ('lin1', nn.Linear(in_dim, 32)),
        ('act1', nn.ReLU()),
        ('lin2', nn.Linear(32, 64)),
        ('act2', nn.ReLU()),
        ('lin3', nn.Linear(64, 16)),
        ('act3', nn.ReLU()),
        ('lin4', nn.Linear(16, out_dim)),
        ('act4', nn.Sigmoid())
    ]))
    model.apply(init_weights)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20])

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_sample_num = 0
        for x, y_true in tqdm(data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            loss = F.mse_loss(y_true, model(x))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_sample_num += x.shape[0]
        print(f"Epoch: {epoch}, Loss: {epoch_loss / epoch_sample_num}")
        lr_scheduler.step()

    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = model(x)
        else:
            Y_pred = torch.cat((Y_pred, model(x)))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)

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

    torch.save(model.state_dict(), f"../ready_models/MTFNN/{node_num}n_co_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")


def mtfnn_msr():
    dataset_path = "../datasets/3c_10w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    M, W = custom_config['M'], custom_config['W']
    Y_train /= W  # Softmax train
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)

    in_dim = M
    out_dim = M
    epochs = 50

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(OrderedDict([
        ('lin1', nn.Linear(in_dim, 8)),
        ('act1', nn.ReLU()),
        ('lin2', nn.Linear(8, 16)),
        ('act2', nn.ReLU()),
        ('lin3', nn.Linear(16, 8)),
        ('act3', nn.ReLU()),
        ('lin4', nn.Linear(8, out_dim)),
        ('act4', nn.Softmax())
    ]))
    model.apply(init_weights)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20])

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_sample_num = 0
        for x, y_true in tqdm(data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            loss = F.mse_loss(y_true, model(x))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_sample_num += x.shape[0]
        print(f"Epoch: {epoch}, Loss: {epoch_loss / epoch_sample_num}")
        lr_scheduler.step()

    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = model(x)
        else:
            Y_pred = torch.cat((Y_pred, model(x)))

    Y_pred *= W
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    scaler_min, scaler_max = custom_config['scaler_min'], custom_config['scaler_max']
    X_test_tensor = X_test_tensor * (scaler_max - scaler_min) + scaler_min
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)

    pred_rate = torch.sum(torch.log2(1.0 + Y_pred * X_test_tensor), dim=1)
    true_rate = torch.sum(torch.log2(1.0 + Y_test_tensor * X_test_tensor), dim=1)

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_cost:\n", pred_rate[:20])
    print("true_cost:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))

    torch.save(model.state_dict(), f"../ready_models/MTFNN/{M}c_smr_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")


class MTFNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MTFNN, self).__init__()
        self.lin1 = nn.Linear(in_dim, 64)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.lin4 = nn.Linear(16, 32)
        self.act4 = nn.ReLU()
        self.lin5 = nn.Linear(32, out_dim)
        self.act51 = nn.Sigmoid()
        self.act52 = nn.Softmax()

    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        x = self.act4(self.lin4(x))
        x = self.lin5(x)
        x[:, :2] = self.act51(x[:, :2])
        x[:, 2:] = self.act52(x[:, 2:])
        return x

def mtfnn_nu():
    width, height = 400, 400
    dataset_path = "../datasets/3u_18mW_10000samples.csv"
    X_train, Y_train, X_test, Y_test, R_test, custom_config = nu_data_load(dataset_path, width, height)
    dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    K, P_sum = custom_config['K'], custom_config['P_sum']

    in_dim = K * 2
    out_dim = 2 + K
    epochs = 100

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MTFNN(in_dim, out_dim)
    model.apply(init_weights)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 60])

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_sample_num = 0
        for x, y_true in tqdm(data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            loss = F.mse_loss(y_true, model(x))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_sample_num += x.shape[0]
        print(f"Epoch: {epoch}, Loss: {epoch_loss / epoch_sample_num}")
        lr_scheduler.step()

    dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True)
    Y_pred = None
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        if Y_pred is None:
            Y_pred = model(x)
        else:
            Y_pred = torch.cat((Y_pred, model(x)))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    for i in range(K):
        X_test_tensor[:, 2 * i] *= width
        X_test_tensor[:, 2 * i + 1] *= height
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)
    Y_test_tensor[:, 0] *= width
    Y_test_tensor[:, 1] *= height
    Y_test_tensor[:, 2:] *= P_sum

    Y_pred[:, 0] *= width
    Y_pred[:, 1] *= height
    Y_pred[:, 2:] *= P_sum
    pred_rate = rate_calc(Y_pred, X_test_tensor)
    true_rate = rate_calc(Y_test_tensor, X_test_tensor)

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    print("Y_pred:\n", Y_pred[:20])
    print("Y_test:\n", Y_test[:20])
    print("X_test:\n", X_test_tensor[:20])
    print("pred_cost:\n", pred_rate[:20])
    print("true_cost:\n", true_rate[:20])
    print(f"less ratio: {torch.sum(pred_rate) / torch.sum(true_rate)}")
    print("avg rate diff:\n", torch.mean(pred_rate - true_rate))

    torch.save(model.state_dict(), f"../ready_models/MTFNN/{K}u_nu_{datetime.datetime.now():%Y%m%d%H%M%S}.pt")


if __name__ == "__main__":
    # mtfnn_co()

    # mtfnn_msr()

    mtfnn_nu()
