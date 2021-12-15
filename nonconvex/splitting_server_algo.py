import numpy as np
import copy
import torch


def peaceman_rachford(z_half, global_model, z_t):
    model = copy.deepcopy(global_model)
    global_model = global_model.state_dict()
    n = len(z_t)

    z_new = copy.deepcopy(z_t)
    for i in range(n):
        for key in z_t[i].keys():
            z_new[i][key] = z_t[i][key] + 2 * (z_half[i][key] - global_model[key])

    global_weights = copy.deepcopy(z_t[0])
    for key in global_weights.keys():
        global_weights[key] = 0 * global_weights[key]

    for key in global_weights.keys():
        for i in range(n):
            global_weights[key] += torch.mul(z_new[i][key], 1/n)
    model.load_state_dict(global_weights)

    temp = copy.deepcopy(z_t[0])
    for key in temp.keys():
        temp[key] = 0 * temp[key]
    local_in_model = []
    for i in range(n):
        local_in_model.append(copy.deepcopy(temp))
    for i in range(n):
        for k in local_in_model[i].keys():
            local_in_model[i][k] = 2*global_weights[k] - z_new[i][k]
    for i in range(n):
        temp = copy.deepcopy(model)
        temp.load_state_dict(local_in_model[i])
        local_in_model[i] = copy.deepcopy(temp)

    return model, local_in_model, z_new


def douglas_rachford(z_half, global_model, z_t, z_half_old):
    """Douglas-Rachfold or aka Partial Inverse"""
    model = copy.deepcopy(global_model)
    global_model = global_model.state_dict()
    n = len(z_t)

    z_new = copy.deepcopy(z_t)
    for i in range(n):
        for key in z_t[i].keys():
            z_new[i][key] = 2*z_half[i][key] - (global_model[key]-z_t[i][key]+z_half_old[i][key])

    global_weights = copy.deepcopy(z_t[0])
    for key in global_weights.keys():
        global_weights[key] = 0 * global_weights[key]

    for key in global_weights.keys():
        for i in range(n):
            global_weights[key] += torch.mul(z_new[i][key], 1/n)
    model.load_state_dict(global_weights)

    temp = copy.deepcopy(z_t[0])
    for key in temp.keys():
        temp[key] = 0 * temp[key]
    local_in_model = []
    for i in range(n):
        local_in_model.append(copy.deepcopy(temp))
    for i in range(n):
        for k in local_in_model[i].keys():
            local_in_model[i][k] = global_weights[k] - z_new[i][k] + z_half[i][k]
    for i in range(n):
        temp = copy.deepcopy(model)
        temp.load_state_dict(local_in_model[i])
        local_in_model[i] = copy.deepcopy(temp)

    return model, local_in_model, z_new



def reflection_projection(z_half, global_model):
    model = copy.deepcopy(global_model)
    global_model = global_model.state_dict()
    n = len(z_half)

    z_new = copy.deepcopy(z_half)
    for i in range(n):
        for key in z_half[i].keys():
            z_new[i][key] = 2*z_half[i][key] - global_model[key]

    global_weights = copy.deepcopy(z_half[0])
    for key in global_weights.keys():
        global_weights[key] = 0 * global_weights[key]
    for key in global_weights.keys():
        for i in range(n):
            global_weights[key] += torch.mul(z_new[i][key], 1/n)
    model.load_state_dict(global_weights)
    local_input_model = [copy.deepcopy(model) for _ in range(n)]
    return model, local_input_model


def fed_average(local_weights, global_model):
    n = len(local_weights)
    # update global weights
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights.keys():
        global_weights[key] = 0 * global_weights[key]
    for key in global_weights.keys():
        for i in range(n):
            global_weights[key] += torch.mul(local_weights[i][key], 1/n)
    global_model.load_state_dict(global_weights)
    local_input_model = [copy.deepcopy(global_model) for _ in range(n)]
    return global_model, local_input_model



def anderson_acceleration(uw, tw, device, args):
    k = len(uw)
    model = copy.deepcopy(uw[0])
    for i in range(k):
        uw[i] = uw[i].state_dict()
        tw[i] = tw[i].state_dict()
    # uw, tw = weights[:-1], weights[1:]
    G = torch.zeros(k, k).to(device)
    for i in range(k):
        for j in range(i, k):
            temp = 0
            for key in uw[0].keys():
                temp += torch.mul(uw[i][key]-tw[i][key], uw[j][key]-tw[j][key]).sum()
            G[i, j] = temp
            G[j, i] = temp

    Ginv = torch.pinverse(G)
    p = torch.matmul(Ginv, torch.ones(k, 1).to(device))
    p = p/p.sum()
    p = (1-args.acc_avg) * p
    p[-1] = p[-1] + args.acc_avg
    # print(p)
    temp = copy.deepcopy(uw[0])
    for key in temp.keys():
        temp[key] = 0 * temp[key]
    final = copy.deepcopy(temp)

    for key in final.keys():
        for j in range(k):
            final[key] += p[j] * tw[j][key]

    model.load_state_dict(final)
    return model

