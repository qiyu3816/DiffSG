"""
Generate dataset for the problem Maximum Sum Rate of Channels using my LRH-Gradient-Descent method (yeah!),
which even performs little better than the exhaustive method with fixed searching step.
"""
import numpy as np
import pandas as pd

from utils.dataset_generate import SUM_RATE_GEN

W = 20.0
M = 80
gs, rates, schemes = SUM_RATE_GEN(sample_num=2000, M=M, W=W)
df = pd.DataFrame(np.concatenate((gs, np.atleast_2d(rates).T, schemes), axis=1))  # rates is the M column
df.to_csv(f"../datasets/{M}c_{int(W)}w.csv", index=False, header=None)
print("Data generation finished.")

# src_csv = pd.read_csv("../datasets/80c_20w_10000samples.csv", header=None)
# data = np.array(src_csv)
# gs, schemes, rates = data[:, :M], data[:, -M:], data[:, M]
# schemes = np.where(schemes < 0, 0.01, schemes)
# line_sum = np.sum(schemes, axis=1)
# diff = line_sum - W
# step_factor = 3  # integer, larger is more feasible
# for i in range(data.shape[0]):
#     for j in range(step_factor * M):
#         while True:
#             index = np.random.randint(0, M)
#             if schemes[i][index] - diff[i] / (step_factor * M) >= 0:
#                 schemes[i][index] -= diff[i] / (step_factor * M)
#                 break
# rates = np.sum(np.log2(1.0 + schemes * gs), axis=1)
# data[:, -M:], data[:, M] = schemes, rates
# print(np.sum(data[:, -M:], axis=1))
# df = pd.DataFrame(data)
# df.to_csv(f"../datasets/{M}c_{int(W)}w_correct.csv", index=False, header=None)
# print("Data correction finished.")

# W = 10.0
# M = 4
#
# src_csv = pd.read_csv("../datasets/4c_10w_src.txt", header=None)
# data = np.array(src_csv)
# gs, rates, schemes = SUM_RATE_GEN(sample_num=1000, M=M, W=W, gs=data[:, :M])
# old_rates = np.sum(np.log2(1.0 + data[:, :M] * data[:, M:2 * M]), axis=1)
# print(np.average(old_rates))
# print(np.average(rates))