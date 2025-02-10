"""
Generate the de-noising trajectory for Computation Offloading.
"""

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from ddpm_opt.diffusion import generate_cosine_schedule
from ddpm_opt.classifier_free_CO import co_data_load, UNet1D, DDPM


@torch.no_grad()
def trajectory_gen_store():
    """
    Load ready model for debug especially.
    """
    T = 20
    omega = 500

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
    diffusion_model = DDPM(T, model, node_num, alphas, device, (1, 3), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ckpts/ddpm_co.pt"))
    diffusion_model.to(device)

    diffusion_model.record_denoise_path = True
    trajectories = np.zeros((X_test.shape[0], Y_test.shape[-1] * diffusion_model.T), dtype=float)
    cur_index = 0
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        _ = diffusion_model.sample(x, omega)
        trajectories[cur_index:cur_index + x.shape[0], :] = diffusion_model.y_i_record
        cur_index += x.shape[0]
    df = pd.DataFrame(trajectories)
    df.to_csv(f"../results/co_denoise_path.csv", header=None, index=False)
    print(f"Trajectory generating finished, {trajectories.shape[0]} samples stored.")


if __name__ == "__main__":
    print("########## Classifier-Free guidance diffusion for Computation Offloading, trajectory generating. ##########")

    trajectory_gen_store()
