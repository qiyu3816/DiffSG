import torch
import torch.nn.functional as F

def class_loss(y_pred, y_true):
    """
    The loss function for CO which adds the classification loss.
    """
    # mse
    mse = torch.mean(torch.square(y_true - y_pred), dim=-1)

    # classification loss
    true_vals = torch.where(torch.lt(y_true, 0.1), 0, 1)
    pred_vals = torch.where(torch.lt(y_pred, 0.1), 0, 1)
    classification_loss = true_vals ^ pred_vals
    classification_loss = torch.sum(classification_loss, dim=-1) * 0.01

    # Loss that penalizes differences between sum(predictions) and sum(labels)
    sum_constraint = torch.square(torch.sum(y_pred, dim=-1) - torch.sum(y_true, dim=-1))
    return torch.sum(mse + classification_loss + sum_constraint, dim=0)

def custom_loss(y_true, y_pred):
    """
    Customized loss function for CO.
    """
    # mse
    mse = torch.mean(torch.square(y_true - y_pred), dim=-1)

    # Loss that penalizes differences between sum(predictions) and sum(labels)
    sum_constraint = torch.square(torch.sum(y_pred, dim=-1) - torch.sum(y_true, dim=-1))

    return torch.sum(mse + sum_constraint)

def vae_loss(y, y_hat, mean, logvar, kld_weight):
    """
    Loss function for VAE.
    """
    reconstruct_loss = F.mse_loss(y_hat, y)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)

    loss = reconstruct_loss + kld_loss * kld_weight
    return loss

def diffusion_opt_loss(estimated_noise, noise, y_t, x_0, alphas, t,
                       lambdas=torch.tensor([0.001, 0.05, 0.05, 0.05, 0.1])):
    """
    :param estimated_noise: estimated noise
    :param noise: source noise
    :param y_t: current y, only resource allocation for simplicity
    :param x_0: raw features ((1, 7) per node)
    :param alphas: de-noising factors
    :param t: positions
    :param lambdas: 5 lagrange multipliers for the loss function
    """
    # L2 loss
    pre_loss = F.mse_loss(estimated_noise, noise)

    # optimization loss
    alpha_cumprod_t = torch.cumprod(alphas, dim=0)[t]
    alpha_cumprod_t = alpha_cumprod_t.repeat(1, y_t.shape[-1]).reshape(y_t.shape[1], y_t.shape[0]).T
    alpha_cumprod_t_1 = torch.cumprod(alphas, dim=0)[t - 1]
    alpha_cumprod_t_1 = alpha_cumprod_t_1.repeat(1, y_t.shape[-1]).reshape(y_t.shape[1], y_t.shape[0]).T
    alphas_t = alphas[t].repeat(1, y_t.shape[-1]).reshape(y_t.shape[1], y_t.shape[0]).T
    y_t_1 = (y_t - (1.0 - alphas_t) / torch.sqrt(1.0 - alpha_cumprod_t_1) * estimated_noise) / torch.sqrt(alphas_t) \
            + (1.0 - alpha_cumprod_t_1) / (1.0 - alpha_cumprod_t) * noise

    F_t = x_0[0][-6]
    kappa = x_0[0][-5]
    P_t = x_0[0][-4]
    P_I = x_0[0][-3]
    B = x_0[0][-2]
    N0 = x_0[0][-1]

    D_y_t = torch.where(y_t > 0.05, 1, 0)  # (batch_size, node_num)
    D_y_t_1 = torch.where(y_t_1 > 0.05, 1, 0)

    s = torch.index_select(x_0, 1, torch.tensor([i for i in range(0, x_0.shape[1] - 6, 7)]))  # (batch_size, node_num)
    c = torch.index_select(x_0, 1, torch.tensor([i for i in range(1, x_0.shape[1] - 6, 7)]))
    w = torch.index_select(x_0, 1, torch.tensor([i for i in range(2, x_0.shape[1] - 6, 7)]))
    theta = torch.index_select(x_0, 1, torch.tensor([i for i in range(3, x_0.shape[1] - 6, 7)]))
    f_l = torch.index_select(x_0, 1, torch.tensor([i for i in range(4, x_0.shape[1] - 6, 7)]))
    h = torch.index_select(x_0, 1, torch.tensor([i for i in range(5, x_0.shape[1] - 6, 7)]))
    alpha = torch.index_select(x_0, 1, torch.tensor([i for i in range(6, x_0.shape[1] - 6, 7)]))

    sinr = P_t * (h ** 2) / (N0 + torch.sum(P_t * (h ** 2)))
    r_u = B * torch.log2(1 + sinr)
    r_d = r_u
    beta = 1.0 - alpha

    # f(D,R)  #############
    # time t cost
    tau_t = torch.where(D_y_t == 1, alpha * (s / r_u + c / (F_t * y_t) + w / r_d), alpha * c / f_l)
    epsilon_t = torch.where(D_y_t == 1, beta * (P_t * s / r_u + P_I * c / (F_t * y_t) + P_t * w / r_d),
                            beta * kappa * (f_l ** 2) * c)  # (batch_size, node_num)
    t_total_cost = torch.sum(tau_t + epsilon_t, dim=1)

    # time t-1 cost
    tau_t_1 = torch.where(D_y_t_1 == 1, alpha * (s / r_u + c / (F_t * y_t_1) + w / r_d), alpha * c / f_l)
    epsilon_t_1 = torch.where(D_y_t_1 == 1, beta * (P_t * s / r_u + P_I * c / (F_t * y_t_1) + P_t * w / r_d),
                            beta * kappa * (f_l ** 2) * c)
    t_1_total_cost = torch.sum(tau_t_1 + epsilon_t_1, dim=1)

    # for simplicity, g_1 can be deleted  #############
    # g_2(D,R)  #############
    delays = torch.where(D_y_t_1 == 1, s / r_u + c / (F_t * y_t_1) + w / r_d, c / f_l)
    g2_tmp = delays - theta
    g2_tmp = torch.where(g2_tmp <= 0, 0.0, g2_tmp)
    g2 = torch.sum(g2_tmp, dim=1)

    # g_3(R)  #############
    g3 = y_t_1 - 1.0
    g3 = torch.where(g3 <= 0, 0.0, g3)
    g3 = torch.sum(g3, dim=1)

    # g_4(R)  #############
    g4 = - y_t_1
    g4 = torch.where(g4 <= 0, 0.0, g4)
    g4 = torch.sum(g4, dim=1)

    # g_5(R)  #############
    g5 = torch.sum(y_t_1, dim=1) - 1.0
    g5 = torch.where(g5 <= 0, 0.0, g5)

    cost_diff = t_1_total_cost - t_total_cost
    cost_diff = torch.where(cost_diff <= 0, 0.0, cost_diff)

    opt_loss = lambdas[0] * cost_diff + lambdas[1] * g2 + lambdas[2] * g3 + lambdas[3] * g4 + lambdas[4] * g5

    return 0.5 * torch.sum(pre_loss) + 0.5 * torch.sum(opt_loss)

def convention_co_opt_loss(y_0, x_0, lambdas=torch.tensor([1.0, 0.05, 0.05, 1.0])):
    """
    :param y_0: the final resource allocation
    :param x_0: conditional features ((1, 3) per node), 0 common features
    :param lambdas: 4 lagrange multipliers for the loss function
    """
    # print(y_0[0])
    y_0 = 1.0 / 2.0 * (y_0 - torch.mean(y_0)) / torch.std(y_0) + 0.5
    y_0 = torch.softmax(y_0, dim=1)

    # D_y_0 = torch.where(y_0 > 0.1, 1, 0)

    local_cost = torch.index_select(x_0, 1, torch.tensor([i for i in range(0, x_0.shape[1], 3)], device=x_0.device))
    offload_transition_cost = torch.index_select(x_0, 1, torch.tensor([i for i in range(1, x_0.shape[1], 3)], device=x_0.device))
    ideal_offload_execution_cost = torch.index_select(x_0, 1, torch.tensor([i for i in range(2, x_0.shape[1], 3)], device=x_0.device))

    # f(D,R)  #############
    # total_cost_t_0 = torch.sum((1 - D_y_0) * local_cost +
    #                            D_y_0 * (offload_transition_cost + ideal_offload_execution_cost / y_0), dim=1)
    total_cost_t_0 = torch.sum(torch.exp(y_0 - 0.1) * local_cost +
                               torch.exp(y_0 - 0.1) * (offload_transition_cost + ideal_offload_execution_cost / y_0), dim=1)

    # for simplicity, g_1 can be deleted  #############
    # no g_2(D,R), only minimize the overall cost  #############

    # g_3(R)  #############
    # g3 = y_0 - 1.0
    # g3 = torch.where(g3 <= 0, 0.0, g3)
    # g3 = torch.sum(g3, dim=1)

    # g_4(R)  #############
    # g4 = - y_0
    # g4 = torch.where(g4 <= 0, 0.0, g4)
    # g4 = torch.sum(g4, dim=1)

    # g_5(R)  #############
    # g5 = torch.sum(y_0, dim=1) - 1.0
    # g5 = torch.where(g5 <= 0, 0.0, g5)

    opt_loss = lambdas[0] * total_cost_t_0 #+ lambdas[3] * g5
    # print("R ", y_0[0])
    # print("cost ", total_cost_t_0)
    # print(f"CONV_CO loss={torch.sum(opt_loss)}")

    return torch.sum(opt_loss)

def sum_rate_loss(p_0, g_0):
    """
    :param p_0: final p, power allocation (batch_size, 3)
    :param g_0: current channel gains (batch_size, 4)
    """
    p_0 = (p_0 - torch.min(p_0)) / (torch.max(p_0) - torch.min(p_0)) * 9.9 + 0.1
    constrain_loss = torch.square((torch.sum(p_0, dim=1) - 10.0))
    no_zero_loss = torch.sum(1.0 / torch.exp(p_0))

    r_0 = torch.sum(torch.log2(1.0 + p_0 * g_0[:, :p_0.shape[1]]), dim=1)
    # print("vectors ", p_t_1, r_t_1, g_0[:, -1])
    # print("p_0 ", p_0)

    # p_0_corrected = torch.zeros_like(p_0)
    # p_0_sum = torch.sum(p_0, dim=1)
    # for i in range(p_0.shape[1]):
    #     p_0_corrected[:, i] = p_0[:, i] - (p_0_sum - 10.0) * (p_0[:, i] / p_0_sum)
    # r_0_corrected = torch.sum(torch.log2(1.0 + p_0_corrected * g_0[:, :3]), dim=1)
    # print("sum of rates ", torch.sum(r_0), torch.sum(r_0_corrected), torch.sum(g_0[:, -1]),
    #                        (torch.sum(g_0[:, -1]) - torch.sum(r_0_corrected)) / r_0_corrected.shape[0])

    # with expert dataset #############
    # opt_loss = torch.square(r_0 - g_0[:, -1])
    # without expert dataset ##########
    opt_loss = -r_0

    # return torch.sum(pre_loss) + torch.sum(opt_loss)
    return torch.sum(constrain_loss) + torch.sum(no_zero_loss) + torch.sum(opt_loss)