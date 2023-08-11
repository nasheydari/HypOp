from src.model import single_node, single_node_xavier
import timeit
from itertools import chain
import torch
from src.timer import Timer
from src.loss import loss_cal_and_update, maxcut_loss_func_helper, loss_maxcut_weighted, loss_sat_weighted, loss_maxind_weighted, loss_maxind_QUBO, loss_maxind_weighted2
from src.utils import mapping_algo, mapping_distribution, gen_q_mis, mapping_distribution_QUBO, get_normalized_G_from_con
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from torch.autograd import grad

def centralized_train(X, G, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}
    X = torch.cat([X[i] for i in X])
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        # embed = nn.Embedding(n, f)
        # embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # # for param in embed.parameters():
        #     param.requires_grad = False
        name=params["model_load_path"]+'conv1_'+file_name[:-4]+'.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"]+'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        # conv2 = single_node(f//2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))
    for i in range(int(params['epoch'])):
        print(i)
        temp = conv1(inputs)
        temp = G @ temp
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp = G @ temp
        temp = torch.sigmoid(temp)
        #temp = torch.softmax(temp, dim=0)
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i==int(params['epoch'])-1:
                name=params["model_save_path"]+'embed_'+file_name[:-4]+'.pt'
                torch.save(embed, name)
                name = params["model_save_path"]+'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"]+'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss=loss

    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    name = './res/plots/Hist_HypOp_QUBO_Maxind/Hist_' + file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2
    # if params['mode'] != 'QUBO':
    #     res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'], params['hyper'])
    # else:
    #     res = mapping_distribution_QUBO(best_out, params, q_torch, n)
    # params2=params
    # params2['Niter_h']=100
    # params2['N_realize'] = 2
    # if params['mode'] != 'QUBO':
    #     res2 = mapping_distribution(best_out, params2, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    # else:
    #     res2 = mapping_distribution_QUBO(best_out, params2, q_torch, n)
    # return res, res2, best_out
    return res, best_out, train_time, map_time

def GD_train(X, G, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}
    X = torch.cat([X[i] for i in X])

    embed = nn.Embedding(n, 1)

    parameters = embed.parameters()
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    inputs = embed.weight
    for i in range(int(params['epoch'])):
        print(i)
        temp = torch.sigmoid(inputs)
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                break
        prev_loss=loss

    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    name = params["plot_path"]+'Hist_' + file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2
    # if params['mode'] != 'QUBO':
    #     res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'], params['hyper'])
    # else:
    #     res = mapping_distribution_QUBO(best_out, params, q_torch, n)
    # params2=params
    # params2['Niter_h']=100
    # params2['N_realize'] = 2
    # if params['mode'] != 'QUBO':
    #     res2 = mapping_distribution(best_out, params2, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    # else:
    #     res2 = mapping_distribution_QUBO(best_out, params2, q_torch, n)
    # return res, res2, best_out
    return res, best_out, train_time, map_time
def get_p(neighbors, probs):
    a = 0
    b = 0
    for nei in neighbors:
        temp_a = 1
        temp_b = 1
        for node in nei:
            temp_a *= probs[node-1]
            temp_b *= (1 - probs[node-1])
        a += temp_a
        b += temp_b
    return a, b

def get_neigh(info):
    neighbors = {}
    for node in info:
        all_edges = info[node]
        temp = []
        for edge in all_edges:
            temp_e = edge.copy()
            temp_e.remove(node)
            temp.append(temp_e)
        neighbors[node] = temp
    return neighbors

def conv(model, x):
    return model(x)

class Trainer:
    def __init__(self, params, f, n, info, log, weights, timer) -> None:
        if params['fast']:
            self.convs1 = [single_node(f, f//2) for x in range(n)]
            self.convs2 = [single_node(f//2, 1) for x in range(n)]
            self.optimizers = []
            for i in range(n):
                parameters = chain(self.onvs1[i].parameters(), self.convs2[i].parameters())
                optimizer = torch.optim.Adam(parameters, lr = params['lr'])
                self.optimizers.append(optimizer)
        else:
            self.convs1 = {x+1: single_node(f, f//2) for x in range(n)}
            self.convs2 = {x+1: single_node(f//2, 1) for x in range(n)}
            self.info = info
            self.optimizers  = {}
            for node in info:
                parameters = chain(self.convs1[node].parameters(), self.convs2[node].parameters())
                self.optimizers[node] = torch.optim.Adam(parameters, lr = params['lr'])
        self.losses = {x: [] for x in info.keys()}
        self.best_outs = {}
        self.timer = timer
        self.epoch = params['epoch']
        self.n = n
        self.params = params
        self.best_loss = float('inf')
        self.weights = weights
        self.log = log
        self.penalty = params['penalty']
        self.hyper = params['hyper']
        self.fixed = {x+1: False for x in range(n)}
        self.patience = params['patience']
        self.tol = 5e-3
        self.ps = {x+1: 0 for x in range(n)}

    def train_fast(self, G, D, Xs, constraints):
        nei = get_neigh(self.info)
        contraints_0 = np.array(constraints) - 1  
        for epoch in range(self.epoch):
            temp_time = timeit.default_timer()
            conv1_res = [conv(self.convs1[i], Xs[i]) for i in range(len(Xs))]
            raw = torch.cat(conv1_res)
            aggre = G @ raw.detach()
            aggre = aggre + raw * D.repeat(self.f//2).view(self.n, self.f//2)
            aggre = torch.relu(aggre)
            self.timer.first_conv_time += (timeit.default_timer() - temp_time)
            temp_time = timeit.default_timer()
            conv2_res = [conv(self.convs2[i], aggre[i]) for i in range(len(aggre))]
            raw = torch.cat(conv2_res)
            aggre = G @ raw.detach()
            aggre = aggre + raw * D
            self.timer.second_conv_time += (timeit.default_timer() - temp_time)
            temp_time = timeit.default_timer()
            aggre = torch.sigmoid(aggre)
            test = aggre.clone().detach().cpu()
            arg_list = [(nei[i+1], test) for i in range(len(nei))]
            with mp.Pool(processes=24) as p:
                predictions = p.starmap(get_p, arg_list) 
            a = torch.tensor([x[0] for x in predictions])
            b = torch.tensor([x[1] for x in predictions])
            loss = maxcut_loss_func_helper(aggre, a, b)
            self.timer.loss_calculate += (timeit.default_timer() - temp_time)
            temp_time = timeit.default_timer()
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.backward(retain_graph=True)
            for optimizer in self.optimizers:
                optimizer.step()
            self.timer.loss_update += (timeit.default_timer() - temp_time)
            temp_time = timeit.default_timer()
            loss_num = loss.detach().item()
            if loss_num < self.best_loss:
                self.best_loss = loss_num
                self.best_outs = aggre.detach().cpu().numpy()
        self.best_outs = {x+1: self.best_outs[x] for x in range(len(self.best_outs))}
        
    def train(self, dcts, Xs, Gs):
        p = 0
        last_loss = {x+1: float('inf') for x in range(self.n)}
        for epoch in range(int(self.epoch)):
            temp_time = timeit.default_timer()
            output = {}
            # convolution 1
            for i in range(1, self.n+1):
                layer = self.convs1[i]
                X = Xs[i]
                output[i] = layer(X)
            aggregated = {}
            for i in range(1, self.n+1):
                input_x = []
                for node in dcts[i]:
                    if node == i:
                        input_x.append(output[i].clone())
                    else:
                        input_x.append(output[node].clone().detach())
                catenated = torch.cat(input_x)
                aggregate = Gs[i] @ catenated
                aggregated[i] = torch.relu(aggregate)
            self.timer.first_conv_time += (timeit.default_timer() - temp_time)
            temp_time = timeit.default_timer()
            
            # convolution 2
            output = {}
            for i in range(1, self.n+1):
                layer = self.convs2[i]
                X = aggregated[i]       
                output[i] = layer(X)
            aggregated = {}
            for i in range(1, self.n+1):
                input_x = []
                for node in dcts[i]:
                    if node == i:
                        input_x.append(output[i].clone())
                    else:
                        input_x.append(output[node].clone().detach())
                catenated = torch.cat(input_x)
                aggregate = Gs[i] @ catenated
                aggregated[i] = aggregate  
                
            self.timer.second_conv_time += (timeit.default_timer() - temp_time)
            args = [(self.optimizers[i], aggregated, self.params, dcts[i], self.weights[i], self.info[i], i, self.timer, self.fixed[i]) for i in range(1, self.n+1)]
            res_pool = [loss_cal_and_update(*arg) for arg in args]
            outs = {}
            sum_loss = 0
            for i, res in enumerate(res_pool):
                sum_loss += res[1]
                outs[i+1] = res[0]
            print(f'At epoch {epoch}, the sum loss is {sum_loss}')
            if sum_loss < self.best_loss:
                p = 0
                print(f'found better loss')
                self.best_outs = outs
                self.best_loss = sum_loss
            else:
                p += 1
                if p > self.patience:
                    print('Early Stopping')
                    break
 #           for i, res in enumerate(res_pool):
 #             sum_loss += res[1]
 #               outs[i+1] = res[0]
 #               if res[1] < last_loss[i+1]:
 #                   self.ps[i+1] = 0
 #                   last_loss[i+1] = res[1]
 #               else:
 #                   self.ps[i+1] += 1
 #                   if self.ps[i+1] > self.patience:
 #                       self.best_outs[i+1] = res[0]

            

    # def map(self, constraints, all_weights, inc):
    #     temp_time = timeit.default_timer()
    #     if self.params['mapping'] == 'trival':
    #         res = {x: 0 if self.best_outs[x] < 0.5 else 1 for x in self.best_outs.keys()}
    #     elif self.params['mapping'] == 'algo':
    #         res = mapping_algo(self.best_outs, self.weights, self.info, self.params['mode'])
    #     else:
    #         res = mapping_distribution(self.best_outs, self.params,
    #                                    self.n, self.info, self.weights,
    #                                    constraints, all_weights, inc, self.penalty, self.hyper,)
    #     mapping_time = timeit.default_timer() - temp_time
    #     self.timer.mapping_time += mapping_time
    #     return res

    def map(self, constraints, all_weights, inc):
        plt.hist(self.best_outs.values(), bins=np.linspace(0, 1, 50))
        #self.best_outs={x: 0.5 for x in self.best_outs}
        #self.best_outs = {x: np.random.uniform(0,1) for x in self.best_outs}
        temp_time = timeit.default_timer()
        if self.params['mapping'] == 'trival':
            res = {x: 0 if self.best_outs[x] < 0.5 else 1 for x in self.best_outs.keys()}
        elif self.params['mapping'] == 'algo':
            res = mapping_algo(self.best_outs, self.weights, self.info, self.params['mode'])
        else:
            res = mapping_distribution(self.best_outs, self.params,
                                       self.n, self.info, self.weights,
                                       constraints, all_weights, inc, self.penalty, self.hyper, )
        mapping_time = timeit.default_timer() - temp_time
        self.timer.mapping_time += mapping_time
        return res
        
