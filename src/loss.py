import torch
import torch.nn.functional as F
import timeit
import numpy as np

def loss_maxcut_weighted(probs, C, dct, weights, hyper=False):
    #print(weights)
    x = probs.squeeze()
    loss = 0
    #print('-----------------')
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        #print(temp_1s, temp_0s, w)
        temp = (temp_1s + temp_0s - 1)
        #print(c, temp)
        loss += (temp * w)
        #print(loss)
    return loss

def loss_maxind_weighted(probs, C, dct, weights):
    p=4
    x = probs.squeeze()
    loss = - sum(x)
    for c, w in zip(C, weights):
        temp = (p * w * x[dct[c[0]]] * x[dct[c[1]]])
        loss += (temp)
    return loss

def loss_maxind_weighted2(probs, C, dct, weights):
    p=4
    x = probs.squeeze()
    loss = - (x.T@x)
    for c, w in zip(C, weights):
        temp = (p * w * x[dct[c[0]]] * x[dct[c[1]]])
        loss += (temp)
    return loss

def loss_sat_weighted(probs, C, dct, weights):
    x = probs.squeeze()
    loss = 0
    for c, w in zip(C, weights):
        temp = 1
        for index in c:
            if index > 0:
                temp *= (1 - x[dct[abs(index)]])
            else:
                temp *= (x[dct[abs(index)]])
        #print(c, temp)
        loss += (temp * w)
    return loss

def loss_cal_and_update(optimizer, aggregated, params, dcts, weights, info, i, timer, fixed): 
    temp_time = timeit.default_timer()
    probs = []
    #print(len(dcts.keys()))
    for node in dcts:
        if node == i:
            probs.append(aggregated[node].clone())
            prob_index_self = len(probs) - 1
        else:
            probs.append(aggregated[node].clone().detach())
    probs = torch.cat(probs).squeeze()
    probs = torch.sigmoid(probs)
    #probs = F.softmax(probs, dim=0)
    if params['mode'] == 'sat':
        loss = loss_sat_weighted(probs, info, dcts, weights)
    elif params['mode'] == 'maxcut':
        loss = loss_maxcut_weighted(probs, info, dcts, weights, params['hyper'])
    elif params['mode'] == 'maxind':
        loss = loss_maxind_weighted(probs, info, dcts, weights)
    timer.loss_calculate += (timeit.default_timer() - temp_time)
    temp_time = timeit.default_timer()
    if fixed:
        res = probs[prob_index_self].clone().detach().item()
        return res, loss.detach().item()
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    #if params['mapping'] == 'trival':
    res = probs[prob_index_self].clone().detach().item()
    #else:
    #    res = aggregated[i].clone().detach().numpy()[0][0]
    timer.loss_update += (timeit.default_timer() - temp_time)
    return res, loss.detach().item()

# for mapping, sat
def loss_sat_numpy(res, C, weights, penalty=0, hyper=True):
    loss = 0
    for c, w in zip(C, weights):
        temp = 1
        for index in c:
            if index > 0:
                temp *= (1 - res[abs(index)])
            else:
                temp *= (res[abs(index)])
        #print(c, temp)
        loss += (temp * w)
    return loss

def loss_sat_numpy_boost(res, C, weights, inc=1.1):
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp = 1
        for index in c:
            if index > 0:
                temp *= (1 - res[abs(index)])
            else:
                temp *= (res[abs(index)])
        #print(c, temp)
        loss += (temp)
        if temp >= 1:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss, new_w

# for mapping, maxcut
def loss_maxcut_numpy(x, C, weights, penalty=0, hyper=False):
    loss = 0
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= (1 - x[index])
                temp_0s *= (x[index])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[index])
                temp_0s *= (x[index])
        temp = (temp_1s + temp_0s - 1)
        #print(c, temp)
        loss += temp
    if loss>-2:
        loss += penalty
    return loss


# for mapping, maxind
def loss_maxind_numpy(x, C, weights, penalty=0, hyper=False):
    p=4
    loss = - sum(x.values())
    for c, w in zip(C, weights):
        temp = p * w * x[c[0]] * x[c[1]]
        loss += (temp)
    return loss


def loss_maxcut_numpy_boost(res, C, weights, inc=1.1):
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        for index in c:
                temp_1s *= (1 - res[index])
                temp_0s *= (res[index])
        temp = (temp_1s + temp_0s - 1)
        loss += (temp)
        if temp >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss, new_w

def loss_maxind_numpy_boost(res, C, weights, inc=1.1):
    p=4
    new_w = []
    loss1 = - sum(res.values())
    loss = - sum(res.values())
    for c, w in zip(C, weights):
        temp = p * w * res[c[0]] * res[c[1]]
        loss += (temp)
        if temp >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss, loss1, new_w




def maxcut_loss_func_helper(X, a, b):
    return X @ a + (1 - X) @ b




def loss_task_weighted(res, C, dct, weights, params, inc=1):
    test = params['test']
    loss = 0
    res = res.squeeze()
    new_w = []
    for c, w in zip(C, weights):
        temp= sum([res[dct[index]] for index in c[2:]])
        if c[0]=='E':
            temp1 = w * torch.relu(c[1] - temp)
        else:
            temp1 = w * torch.relu(temp - c[1])
        loss += (temp1)
        if temp1 >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss


def loss_task_numpy(x, C, weights, params, penalty=0, hyper=True):
    inc=1
    test=params['test']
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp = sum([x[index] for index in c[2:]])
        if c[0] == 'E':
            temp1 = w * max(0, c[1] - temp)
        else:
            temp1 = w * max(0, temp - c[1])
        loss += (temp1)
        if temp1 >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    loss += (test * sum(x))**2
    return loss


def loss_task_numpy_boost(x, C, weights, params, penalty=0, hyper=True, inc=1.1):
    test=params['test']
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp = sum([x[index] for index in c[2:]])
        if c[0] == 'E':
            temp1 = w * max(0, c[1] - temp)
        else:
            temp1 = w * max(0, temp - c[1])
        loss += (temp1)
        if temp1 >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    loss += (test * sum(x))**2
    return loss


def loss_maxind_QUBO(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    #probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    cost = (probs.T @ Q_mat @ probs).squeeze()

    return cost