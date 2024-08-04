import torch
import torch.nn.functional as F
import timeit
import numpy as np

def loss_maxcut_weighted(probs, C, weights, penalty_inc, penalty_c,  hyper):
    #print(weights)
    x = probs.squeeze()
    loss = 0
    penalty=0
    #print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            # indices = [i - 1 for i in c]
            # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
            for index in c:
                temp_1s *= (1 - x[index-1])
                temp_0s *= (x[index-1])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[index-1])
                temp_0s *= (x[index-1])
        temp = (temp_1s + temp_0s)

        #print(temp_1s, temp_0s, w)
        #temp = (temp_1s + temp_0s - 1)

        #print(c, temp)
        loss += (temp * w)
        #print(loss)
    if penalty_inc:
        penalty = torch.sum(torch.min((1 - x), x))
        loss += penalty_c * penalty
    return loss


def loss_maxcut_weighted_coarse(probs, C, dct, weights, hyper=False):
    x = probs.squeeze()
    loss = 0
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:#need changes if hyper
            for index in c:
                temp_1s *= 1 - x[dct[index]]
                temp_0s *= x[dct[index]]
        else:
            for index in c[0:2]:
                temp_1s *= 1 - x[dct[index][0]-1][dct[index][1]]
                temp_0s *= x[dct[index][0]-1][dct[index][1]]
        temp = temp_1s + temp_0s
        loss += temp * w
    return loss


def loss_maxcut_weighted_multi_gpu(probs, C, dct, weights_tensor, hyper=False, TORCH_DEVICE='cpu'):
    x = probs.squeeze()
    loss = 0
    #weights_tensor = torch.tensor(weights, dtype=torch.float32)
    #pdb.set_trace()
    temp_values = torch.zeros(len(C)).to(TORCH_DEVICE)
    for idx, c in enumerate(C):
        if hyper:
            indices = [dct[index] for index in c]
        else:
            indices = [dct[index] for index in c[0:2]]
        
        # Using advanced indexing for acceleration
        selected_x = x[indices]
        temp_1s = torch.prod(1 - selected_x)
        temp_0s = torch.prod(selected_x)
        temp_values[idx] = temp_1s + temp_0s - 1
    loss = torch.sum(temp_values * weights_tensor)
    return loss

def loss_maxcut_weighted_multi(
    probs,
    C,
    dct,
    weights_tensor,
    hyper=False,
    TORCH_DEVICE="cpu",
    outer_constraint=None,
    temp_reduce=None,
    start=0,
):
    x = probs.squeeze()
    loss = 0
    inner_temp_values = torch.zeros(len(outer_constraint) + len(C)).to(TORCH_DEVICE)
    out_point = len(C)
    total_C = C + outer_constraint
    for idx, c in enumerate(total_C):
        if hyper:
            indices = [dct[index] for index in c]
        else:
            indices = [dct[index] for index in c[0:2]]
        """
        indices = [index-start for index in indices]
        selected_x = x[indices]
        temp_1s = torch.prod(1 - selected_x)
        temp_0s = torch.prod(selected_x)
        inner_temp_values[idx] =  temp_1s + temp_0s - 1
        """
        indices = [index - start + 1 for index in indices]
        if idx < out_point:
            selected_x = x[indices]
            temp_1s = torch.prod(1 - selected_x)
            temp_0s = torch.prod(selected_x)
            inner_temp_values[idx] = temp_1s + temp_0s - 1
        else:
            res = [
                x[indice]
                if indice >= 0 and indice < len(x)
                else temp_reduce[indice + start - 1]
                for indice in indices
            ]
            selected_x = torch.stack(res)
            temp_1s = torch.prod(1 - selected_x)
            temp_0s = torch.prod(selected_x)
            inner_temp_values[idx] = 0.5 * (temp_1s + temp_0s - 1)

    loss = torch.sum(inner_temp_values * weights_tensor)
    return loss




def loss_maxcut_weighted_anealed(probs, C, dct, weights, temper, hyper=False):
    x = probs.squeeze()
    loss = 0
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= 1 - x[dct[index]]
                temp_0s *= x[dct[index]]
        else:
            for index in c[0:2]:
                temp_1s *= 1 - x[dct[index]]
                temp_0s *= x[dct[index]]
        temp = temp_1s + temp_0s
        loss += temp * w
    Entropy = sum(
        [item * torch.log2(item) + (1 - item) * torch.log2(1 - item) for item in x]
    )
    loss += temper * Entropy
    return loss




def loss_mincut_weighted(probs, C, weights, penalty_inc, penalty_c, indicest, hyper):
    #print(weights)
    x = probs.squeeze()
    n=x.shape[0]
    loss = 0
    penalty=0
    #print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            # indices = [i - 1 for i in c]
            # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
            for index in c:
                temp_1s *= (1 - x[index-1])
                temp_0s *= (x[index-1])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[index-1])
                temp_0s *= (x[index-1])
        temp = (temp_1s + temp_0s)-1


        loss += (temp * w)
        #print(loss)
    loss2=(2*torch.sum(x)-n)**2-loss
    if penalty_inc:
        penalty = torch.sum(torch.min((1 - x), x))
        loss2 += penalty_c * penalty
    return loss2




def loss_partitioning_weighted(res, C, weights, hyper):
    #print(weights)
    x_c = res.squeeze()
    [n, m] = x_c.shape


    loss = 0
    temp=0
    penalty=0
    #print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp=0
        temp_1s = 1
        temp_0s = 1
        for col in range(m):
            if hyper:
                # indices = [i - 1 for i in c]
                # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
                    for index in c:
                        temp_1s *= (1 - x_c[index-1,col])
                        temp_0s *= (x_c[index-1,col])
            else:
                for index in c[0:2]:
                    temp_1s *= (1 - x_c[index-1,col])
                    temp_0s *= (x_c[index-1,col])
            temp += (temp_1s + temp_0s)-1


        loss += (temp * w)
        #print(loss)
    loss2=sum((m*torch.sum(x_c, axis=0)-n)**2)-loss
    # if penalty_inc:
    #     penalty = torch.sum(torch.min((1 - x_c), x_c))
    #     loss2 += penalty_c * penalty
    return loss2


def loss_partitioning_nonbinary(res,  C, m, weights, hyper):
    #print(weights)
    x_c = res.squeeze()
    n = x_c.shape[0]


    loss = 0
    temp=0
    penalty=0
    #print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp=0
        temp_1s = 1
        temp_0s = 1
        if hyper:
            avg=sum([x_c[cj-1] for cj in c])/len(c)
            # indices = [i - 1 for i in c]
            # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
            for index in c:
                temp += (x_c[index-1]-avg)**2
        else:
            avg = sum([x_c[cj - 1] for cj in c[0:2]]) / len(c)
            for index in c[0:2]:
                temp += (x_c[index-1]-avg)**2


        loss += (temp * w)
        #print(loss)
    loss2=0
    for col in range(m):
        loss2+=(m*(n-sum([min((col-x_c[i])**2,1) for i in range(n)]))-n)**2
    loss2+=-loss
    # if penalty_inc:
    #     penalty = torch.sum(torch.min((1 - x_c), x_c))
    #     loss2 += penalty_c * penalty
    return loss2

def loss_partition_numpy(res, C, weights, hyper):
    # print(weights)
    x_c = res
    [n, m] = x_c.shape

    loss = 0
    temp = 0
    penalty = 0
    # print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp = 0
        temp_1s = 1
        temp_0s = 1
        for col in range(m):
            if hyper:
                # indices = [i - 1 for i in c]
                # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
                for index in c:
                    temp_1s *= (1 - x_c[index - 1, col])
                    temp_0s *= (x_c[index - 1, col])
            else:
                for index in c[0:2]:
                    temp_1s *= (1 - x_c[index - 1, col])
                    temp_0s *= (x_c[index - 1, col])
            temp += (temp_1s + temp_0s)-1

        loss += (temp * w)
        # print(loss)
    loss2 = sum((m*np.sum(x_c, axis=0)-n)**2)-loss
    # if penalty_inc:
    #     penalty = torch.sum(torch.min((1 - x_c), x_c))
    #     loss2 += penalty_c * penalty
    return loss2


def loss_partition_numpy_boost(res, C, weights, L, hyper):
    # print(weights)
    x_c = res

    n=res.shape[0]
    loss = 0
    temp = 0
    penalty = 0
    # print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp = 0
        temp_1s = 1
        temp_0s = 1
        for col in range(L):
            if hyper:
                # indices = [i - 1 for i in c]
                # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
                for index in c:
                    temp_1s *= (1 - x_c[index - 1, col])
                    temp_0s *= (x_c[index - 1, col])
            else:
                for index in c[0:2]:
                    temp_1s *= (1 - x_c[index - 1, col])
                    temp_0s *= (x_c[index - 1, col])
            temp += (temp_1s + temp_0s)-1

        loss += (temp * w)
        # print(loss)
    loss2 = (L*np.sum(x_c, axis=0)-n)**2
    # if penalty_inc:
    #     penalty = torch.sum(torch.min((1 - x_c), x_c))
    #     loss2 += penalty_c * penalty
    return loss2, -loss


def loss_maxcut_weighted_anealed(probs, C, dct, weights,  temper, hyper=False):
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
        #temp = (temp_1s + temp_0s - 1)
        temp = (temp_1s + temp_0s)
        #print(c, temp)
        loss += (temp * w)
        #print(loss)
    Entropy=sum([item* torch.log2(item)+(1-item)*torch.log2(1-item) for item in x])
    loss+= temper*Entropy
    return loss


def loss_task_weighted(res, C, dct, weights):
    loss = 0
    x = res.squeeze()
    #weight2=torch.ones((len(res),))
    #weight2[0:int(np.floor(len(res)/2))]=2*weight2[0:int(np.floor(len(res)/2))]
    new_w = []
    for c, w in zip(C, weights):
        temp=sum([x[dct[index]] for index in c[:-1]])
        #temp= torch.sum(torch.tensor([x[dct[index]] for index in c[:-1]], requires_grad=True))
        if c[-1]=='T':
            temp1 = w * max(np.ceil((len(c)-1)/2)-temp , 0)
            #temp1 = w * torch.relu(torch.floor(torch.tensor(len(c)/2))-temp)
        else:
            temp1 = w * max(temp - np.ceil((len(c)-1)/2) , 0 )
            #temp1 = w * torch.relu(temp - torch.ceil(torch.tensor(len(c)/2)))
        loss = loss+temp1
    #temp3=sum([weight2[index]*x[index] for index in range(len(x))])
    temp2 = 0.2*sum([min(1 - x[index], x[index]) for index in range(len(x))])
    #loss = loss+temp2+temp3
    loss = loss + temp2
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
        loss = loss_maxcut_weighted(probs, info, weights, params['penalty_inc'], params['penalty_c'], params['hyper'])
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
    # if hyper:
    #     loss=sum([np.prod([x[index] for index in c]) + np.prod([1 - x[index] for index in c]) - 1 for c in C])
    # else:
    #     loss=sum([np.prod([x[index] for index in c[0:2]]) + np.prod([1 - x[index] for index in c[0:2]]) - 1 for c in C])
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
    return loss

def loss_mincut_numpy(x, C, weights, penalty=0, hyper=False):
    # if hyper:
    #     loss=sum([np.prod([x[index] for index in c]) + np.prod([1 - x[index] for index in c]) - 1 for c in C])
    # else:
    #     loss=sum([np.prod([x[index] for index in c[0:2]]) + np.prod([1 - x[index] for index in c[0:2]]) - 1 for c in C])
    loss = 0
    n=len(x)
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
    loss2 = (2 * sum(x.values()) - n) ** 2 - loss
    return loss2




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

def loss_mincut_numpy_boost(res, C, weights, inc=1.1):
    loss = 0
    new_w = []
    n=len(res)
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
    loss2=(2 * sum(res.values()) - n)
    return loss2, -loss, new_w

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






def loss_task_numpy(res, C, weights,  penalty=0, hyper=False):
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp= sum([res[index] for index in c[:-1]])
        if c[-1]=='T':
            temp1 = w * max(np.ceil((len(c)-1)/2)-temp,0)
        else:
            temp1 = w * max(temp - np.ceil((len(c)-1)/2),0)
        loss += (temp1)
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


def loss_task_weighted_vec(res, lenc, leninfo):
    x_c = res.squeeze()
    [n,m]=x_c.shape
    temp=lenc-torch.sum(x_c,0)
    temp1 = torch.maximum(temp, torch.zeros([m]))
    temp1s=sum(temp1)
    temp2=torch.sum(x_c,1)-leninfo-50*torch.ones([n])
    temp3=torch.maximum(temp2, torch.zeros([n]))
    temp3s=sum(temp3)
    loss = temp1s + temp3s
    return loss

#def loss_task_weighted_vec(res, C, C_dic,  n, info, weights):
    # loss = 0
    # x_c = res.squeeze()
    # temp1 = 0
    # temp3 = 0
    # for c, w in zip(C, weights):
    #     temp = sum(x_c[:,C_dic[str(c)]])
    #     #temp= torch.sum(torch.tensor([x[dct[index]] for index in c[:-1]], requires_grad=True))
    #     temp1 = temp1 + w * max(len(c)-temp , 0)
    #     #temp1 = w * torch.relu(torch.floor(torch.tensor(len(c)/2))-temp)
    # for i in range(n):
    #     temp2 = sum(x_c[i, :])
    #     temp3 = temp3 + max(temp2 - len(info[i+1])-1, 0 )
    # #temp3=sum([weight2[index]*x[index] for index in range(len(x))])
    # #loss = loss+temp2+temp3
    # loss =temp1 + temp3
    # return loss

def loss_task_numpy_vec(res, lenc, leninfo):
    # x_c=np.array(list(res.values()))
    x_c=res
    [n,m]=x_c.shape
    temp=lenc-np.sum(x_c,axis=0)
    temp1 = np.maximum(temp, np.zeros([m,]))
    temp1s=sum(temp1)
    temp2=np.sum(x_c,axis=1)-leninfo-50*np.ones([n])
    temp3=np.maximum(temp2, np.zeros([n,]))
    temp3s=sum(temp3)
    loss = temp1s + temp3s
    return loss

# def loss_task_numpy_vec(res, C, C_dic, n, info, weights):
#     loss = 0
#     x_c=np.array(list(res.values()))
#     temp1 = 0
#     temp3 = 0
#     for c, w in zip(C, weights):
#         temp = sum(x_c[:,C_dic[str(c)]])
#         #temp= torch.sum(torch.tensor([x[dct[index]] for index in c[:-1]], requires_grad=True))
#         temp1 = temp1 + w * max(len(c)-temp , 0)
#         #temp1 = w * torch.relu(torch.floor(torch.tensor(len(c)/2))-temp)
#     for i in range(n):
#         temp2 = sum(x_c[i, :])
#         temp3 = temp3 + max(temp2 - len(info[i+1])-1 , 0 )
#     #temp3=sum([weight2[index]*x[index] for index in range(len(x))])
#     #loss = loss+temp2+temp3
#     loss =temp1 + temp3
#     return loss



def loss_maxind_QUBO_coarse(probs, Q_mat, dct, n):

    x=torch.zeros([n,])
    for i in range(n):
        x[i]=probs[dct[i+1][0] - 1][dct[i+1][1]]
    #probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x

    cost = (x.T @ Q_mat @ x).squeeze()

    return cost

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


def loss_watermark(res_wat,watermark_cons, wat_type):
    p = 4
    loss = - sum(res_wat.values())
    for c in watermark_cons:
        temp = p * (res_wat[c[0]] * res_wat[c[1]]) + p * (1-res_wat[c[0]]) * (1-res_wat[c[1]])
        loss += (temp)
    return loss


def loss_MNP_weighted(res, C, weights, hyper):
    #print(weights)
    x_c = res.squeeze()
    [n, m] = x_c.shape


    loss = 0
    temp=0
    penalty=0
    #print('-----------------')
    # loss=sum([torch.prod(x[indicest[i]]) + torch.prod(1 - x[indicest[i]]) for i in range(len(C))])
    for c, w in zip(C, weights):
        temp=0
        temp_1s = 1
        temp_0s = 1
        for col in range(m):
            if hyper:
                # indices = [i - 1 for i in c]
                # temp=torch.prod(x[indices])+torch.prod(1-x[indices])
                    for index in c:
                        temp_1s *= (1 - x_c[index-1,col])
                        temp_0s *= (x_c[index-1,col])
            else:
                for index in c[0:2]:
                    temp_1s *= (1 - x_c[index-1,col])
                    temp_0s *= (x_c[index-1,col])
            temp += (temp_1s + temp_0s)-1


        loss += (temp * w)
        #print(loss)
    loss2=sum((m*torch.sum(x_c, axis=0)-n)**2)-loss
    # if penalty_inc:
    #     penalty = torch.sum(torch.min((1 - x_c), x_c))
    #     loss2 += penalty_c * penalty
    return loss2