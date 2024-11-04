"""
Generate dataset for the problem NOMA-UAV by hierarchical solving.
"""

from tqdm import tqdm
import numpy as np
import pandas as pd


def coordinates_gen(sample_num, K=3, width=400, height=400):
    """
    Assume that K=3 is fixed.
    """
    qs = np.zeros((sample_num, 6))
    for i in range(sample_num):
        blocks = np.array([0, 0, 0, 0])
        for j in range(K):
            b = np.random.choice(np.where(blocks == 0)[0])
            blocks[b] = 1
            x = np.random.randint(width // 2 * (b % 2) + 1, width // 2 * (1 + b % 2) + 1)
            y = np.random.randint(height // 2 * (b // 2) + 1, height // 2 * (1 + b // 2) + 1)
            qs[i][j * 2], qs[i][j * 2 + 1] = x, y
    return qs


def feasible_solution(P_sum):
    """
    Generate feasible solutions satisfying the sorting order.
    :return: solutions(solution_num, K)
    """
    K = 3
    step = 0.1
    solutions = None
    i_cycle = np.arange(P_sum / 3 + step, P_sum - 2 * step, step)
    for i in i_cycle:
        j_cycle = np.arange((P_sum - i) / 2 + step, P_sum - i - step, step)
        for j in j_cycle:
            k = P_sum - i - j
            cur_solution = np.array([k, j, i])
            if solutions is None:
                solutions = np.atleast_2d(cur_solution)
            else:
                solutions = np.concatenate((solutions, np.atleast_2d(cur_solution)))
    return solutions


def is_point_inside_triangle(a, b, c, d):
    """
    If point a in triangle bcd, return True.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(a, b, c)
    d2 = sign(a, c, d)
    d3 = sign(a, d, b)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def noma_uav_gen(sample_num, P_sum):
    fs = feasible_solution(P_sum)  # smaller->larger order
    sigma_sq = 110
    rou_0 = 60
    H = 150
    K = 3
    width, height = 400, 400

    qs = coordinates_gen(sample_num, K=K, width=width, height=height)
    step = 1
    x_L, y_L = np.arange(0, width + step, step), np.arange(0, height + step, step)
    X, Y = np.meshgrid(x_L, y_L)
    X, Y = X.flatten(), Y.flatten()

    data = np.zeros((sample_num, 2 * K + 2 + K + 1))  # uav coordinates, L coordinate, Pi, final rate
    for i in tqdm(range(sample_num)):
        # generate a sample in this cycle
        round_solutions = None
        for j in range(X.shape[0]):
            if not is_point_inside_triangle([X[j], Y[j]], qs[i, 0:2], qs[i, 2:4], qs[i, 4:]):
                continue

            # generate a solution with given UAV coordinate in this cycle
            h = np.zeros(K)
            for jj in range(K):
                h[jj] = np.sqrt(rou_0 / (H ** 2 + (X[j] - qs[i][jj * 2]) ** 2 + (Y[j] - qs[i][jj * 2 + 1]) ** 2))
            # get feasible solutions with given h
            sorted_indices = np.argsort(-h)
            si = np.zeros(K, dtype=int)
            for ii, jj in enumerate(sorted_indices):
                si[jj] = ii
            feasible_solutions = fs[:, si]

            sinr = np.zeros_like(feasible_solutions)
            for jj, kk in enumerate(sorted_indices):
                if jj == 0:
                    sinr[:, kk] = feasible_solutions[:, kk] * np.square(h[kk]) / sigma_sq
                else:
                    sinr[:, kk] = feasible_solutions[:, kk] / (np.sum(feasible_solutions[sorted_indices[:jj]]) + sigma_sq / np.square(h[kk]))
            rates = np.sum(np.log2(1 + sinr), axis=1)
            solution_index = np.argmax(rates)
            if round_solutions is None:
                round_solutions = np.concatenate((np.array([X[j], Y[j]]), feasible_solutions[solution_index], np.array([rates[solution_index]])))
                round_solutions = np.atleast_2d(round_solutions)
            else:
                tmp = np.atleast_2d(np.concatenate((np.array([X[j], Y[j]]), feasible_solutions[solution_index], np.array([rates[solution_index]]))))
                round_solutions = np.concatenate((round_solutions, tmp))
        if round_solutions is not None:
            final_solution_index = np.argmax(round_solutions[:, -1])
            data[i, :2 * K], data[i, 2 * K:] = qs[i], round_solutions[final_solution_index]
        else:
            print(i)
    return data


def rotate_point(point, center, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    x, y = point
    cx, cy = center

    # 应用旋转矩阵
    x_rotated = np.cos(angle_radians) * (x - cx) - np.sin(angle_radians) * (y - cy) + cx
    y_rotated = np.sin(angle_radians) * (x - cx) + np.cos(angle_radians) * (y - cy) + cy

    return x_rotated, y_rotated


def dataset_extension(dataset_path):
    """
    Extend current dataset by translating, flipping, rotating.
    """
    src_csv = pd.read_csv(dataset_path, header=None)
    src_data = np.array(src_csv)
    times = 3
    width, height = 400, 400
    rotation_angle_upper = 10

    extended_data = np.zeros((src_data.shape[0] * times, src_data.shape[1]))
    for i in range(times):
        for j in range(src_data.shape[0]):
            method = np.random.randint(2)
            cur_index = i * src_data.shape[0] + j
            if method == 0:  # translate
                x_min, x_max = np.min(src_data[j][[0, 2, 4]]), np.max(src_data[j][[0, 2, 4]])
                y_min, y_max = np.min(src_data[j][[1, 3, 5]]), np.max(src_data[j][[1, 3, 5]])
                x_diff = (np.random.randint(width - x_max) if width > x_max else 0) - x_min
                y_diff = (np.random.randint(height - y_max) if height > y_max else 0) - y_min
                extended_data[cur_index] = src_data[j]
                extended_data[cur_index][[0, 2, 4, 6]] += x_diff
                extended_data[cur_index][[1, 3, 5, 7]] += y_diff
            else:  # flip
                extended_data[cur_index] = src_data[j]
                extended_data[cur_index][[0, 2, 4, 6]] = width - src_data[j][[0, 2, 4, 6]]
                extended_data[cur_index][[1, 3, 5, 7]] = height - src_data[j][[1, 3, 5, 7]]

                triangle_points = np.array([[extended_data[cur_index][0], extended_data[cur_index][1]],
                                            [extended_data[cur_index][2], extended_data[cur_index][3]],
                                            [extended_data[cur_index][4], extended_data[cur_index][5]]])
                inside_point = np.array([extended_data[cur_index][6], extended_data[cur_index][7]])
                centroid = np.mean(triangle_points, axis=0)
                rotation_angle = np.random.randint(low=-rotation_angle_upper, high=rotation_angle_upper)
                rotated_triangle = np.array([rotate_point(p, centroid, rotation_angle) for p in triangle_points])
                rotated_inside_point = rotate_point(inside_point, centroid, rotation_angle)
                extended_data[cur_index, 0], extended_data[cur_index, 2], extended_data[cur_index, 4] = rotated_triangle[0, 0], rotated_triangle[1, 0], rotated_triangle[2, 0]
                extended_data[cur_index, 1], extended_data[cur_index, 3], extended_data[cur_index, 5] = rotated_triangle[0, 1], rotated_triangle[1, 1], rotated_triangle[2, 1]
                extended_data[cur_index, 6], extended_data[cur_index, 7] = rotated_inside_point[0], rotated_inside_point[1]
    return extended_data


if __name__ == "__main__":
    sample_num = 2500
    P_sum = 18
    # data = noma_uav_gen(sample_num, P_sum)
    # df = pd.DataFrame(data)
    # df.to_csv(f"../datasets/3u_{int(P_sum)}mW_{sample_num}samples.csv", index=False, header=None)

    extended_data = dataset_extension("../datasets/3u_18mW_2500samples.csv")
    df = pd.DataFrame(extended_data)
    df.to_csv(f"../datasets/3u_{int(P_sum)}mW_extension.csv", header=None, index=False)
