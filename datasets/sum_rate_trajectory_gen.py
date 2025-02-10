"""
Generate the de-noising trajectory for Maximum Sum Rate of Channels.
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

from ddpm_opt.diffusion import generate_cosine_schedule
from ddpm_opt.UNetCF import UNet1D
from ddpm_opt.classifier_free_MSR import DDPM, msr_data_load, custom_decoder


@torch.no_grad()
def trajectory_gen_store():
    """
    Load ready model for debug especially.
    """
    T = 20
    omega = 500

    dataset_path = "../datasets/3c_10w_10000samples.csv"
    X_train, Y_train, X_test, Y_test, custom_config = msr_data_load(dataset_path)
    M, W = custom_config['M'], custom_config['W']

    alphas = 1.0 - generate_cosine_schedule(T)

    model = UNet1D(input_dim=M, proj_dim=128, cond_dim=custom_config['sfn'] * M,  # + custom_config['cdim'],
                   dims=(64, 32, 16, 8), is_attn=(False, False, False, False), middle_attn=False, n_blocks=2)

    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = DDPM(T, model, M, W, alphas, device, (1, M), custom_config, 0.1,
                           0.9999, 10, 5, False)
    diffusion_model.load_state_dict(torch.load("../ckpts/ddpm_msr_3c.pt"))
    diffusion_model.to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    diffusion_model.record_denoise_path = True
    _ = diffusion_model.sample(X_test_tensor, omega)
    df = pd.DataFrame(diffusion_model.y_i_record)
    df.to_csv(f"../results/msr_denoise_path.csv", header=None, index=False)
    print(f"Trajectory generating finished, {diffusion_model.y_i_record.shape[0]} samples stored.")


if __name__ == "__main__":
    print("########## Classifier-Free guidance diffusion for MSR, trajectory generating. ##########")

    trajectory_gen_store()
