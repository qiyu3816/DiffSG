# DiffSG: Diffusion for Solution Generation in Network Optimization

## Introduction

This repository contains the code and data used to study diffusion models as supervised solution generators for network optimization problems. Given the parameters of an optimization instance as the condition, a classifier-free guidance DDPM generates the corresponding solution vector.

Three tasks are included:

- **CO**: computation offloading (`classifier_free_CO.py`)
- **MSR**: maximum sum-rate power allocation (`classifier_free_MSR.py`)
- **NU**: NOMA-UAV placement and power allocation (`classifier_free_NU.py`)

There is no single global `main.py`. Each file above contains the data loader, task-specific DDPM wrapper, training function, evaluation function, and script entry point for one task.

## Repository structure

- **`ddpm_opt/`**: task entry scripts, the 1-D conditional U-Net, diffusion utilities, and EMA helper.
- **`datasets/`**: ready-made CSV datasets and dataset-generation scripts.
- **`ckpts/`**: pretrained DDPM and baseline checkpoints.
- **`utils/`**: preprocessing and experimental/legacy loss utilities.
- **`baselines/`**: implementations of the baselines used in the article.
- **`results/`**: result-processing scripts and denoising trajectories.

## Environment

The main DDPM scripts require Python and the following packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pandas tqdm scikit-learn
```

CUDA is used automatically when available; otherwise the scripts run on CPU.

## Quick start: evaluate the pretrained models

The paths in the original scripts are relative to `ddpm_opt/`, so run the following commands from that directory:

```bash
cd /path/to/DiffSG/ddpm_opt
```

Computation offloading:

```bash
python classifier_free_CO.py
```

Maximum sum rate (3 channels):

```bash
python classifier_free_MSR.py
```

NOMA-UAV (3 users):

```bash
python -c "from classifier_free_NU import load_test_nu; load_test_nu('../ckpts/ddpm_nu_3u.pt')"
```

The CO and MSR files call their standard evaluation functions in `__main__`. The current `classifier_free_NU.py` entry point calls `load_test_nu_debug()` instead, which generates `results/nu_denoise_path.csv`; the one-line command above calls the normal NU evaluation function.

The supplied task configurations are:

| Task | Dataset | Pretrained checkpoint |
| --- | --- | --- |
| CO | `datasets/3nodes_50000samples_new.csv` | `ckpts/ddpm_co.pt` |
| MSR | `datasets/3c_10w_10000samples.csv` | `ckpts/ddpm_msr_3c.pt` |
| NU | `datasets/3u_18mW_10000samples.csv` | `ckpts/ddpm_nu_3u.pt` |

## Training

Training functions are present in the same three task scripts. They use the hard-coded dataset and hyperparameter settings in those functions and run for 200 epochs by default. The following commands train a model and save a new checkpoint:

```bash
# Computation offloading
python -c "import torch; from classifier_free_CO import train_ddpm_co; m=train_ddpm_co(); torch.save(m.state_dict(), '../ckpts/ddpm_co_local.pt')"

# Maximum sum rate
python -c "import torch; from classifier_free_MSR import train_ddpm_msr; m=train_ddpm_msr(); torch.save(m.state_dict(), '../ckpts/ddpm_msr_3c_local.pt')"

# NOMA-UAV
python -c "import torch; from classifier_free_NU import train_ddpm_nu; m=train_ddpm_nu(); torch.save(m.state_dict(), '../ckpts/ddpm_nu_3u_local.pt')"
```

Alternatively, uncomment the corresponding training and `torch.save(...)` lines in the `if __name__ == "__main__":` block of the task script.

## How the main modules are connected

The three task scripts have the same high-level pipeline:

1. **Data loading.** `co_data_load`, `msr_data_load`, or `nu_data_load` reads a CSV with pandas, separates the instance features `X` from the target solution `Y`, applies task-specific normalization, and takes the first 70% as training data and the last 30% as test data. The arrays are wrapped in `TensorDataset` and `DataLoader`.
2. **Noise schedule and model.** `generate_cosine_schedule` from `ddpm_opt/diffusion.py` creates the diffusion schedule. `UNet1D` from `ddpm_opt/UNetCF.py` receives the noisy solution `y_t`, normalized time step `t/T`, condition `X`, and a condition mask. Its residual blocks inject both time and condition embeddings and use U-Net skip connections to predict the added noise.
3. **Forward diffusion and loss.** During training, the task-specific `DDPM.forward(Y, X)` samples a random time step and Gaussian noise, constructs `y_t` from the clean solution `Y`, randomly drops some conditions for classifier-free training, and calls `UNet1D`. The training loss is `F.mse_loss(predicted_noise, noise)`, followed by the usual Adam update.
4. **Reverse sampling.** During evaluation, `DDPM.sample(X, omega)` starts from Gaussian noise and iterates from `T-1` to `0`. At every step it evaluates the U-Net once without the condition and once with the condition, combines the two noise estimates using classifier-free guidance, and applies the DDPM reverse update.
5. **Decoding and metrics.** The raw generated vector is converted to a feasible task solution by the task-specific decoder (for example, softmax-based resource/power normalization), then compared with the test solution using cost, sum rate, or accuracy metrics.

In short, the active path is:

```text
CSV -> task data loader -> DataLoader -> task DDPM wrapper -> UNet1D
    -> noise-prediction MSE (training)
    -> classifier-free reverse sampling -> task decoder/metric (evaluation)
```

`ddpm_opt/diffusion.py` also contains an earlier generic `DiffusionOpt` implementation, and `utils/loss_function.py` contains experimental optimization-aware losses. They are not called by the current `classifier_free_CO.py`, `classifier_free_MSR.py`, or `classifier_free_NU.py` training paths; the active loss in these scripts is the noise-prediction MSE described above.
