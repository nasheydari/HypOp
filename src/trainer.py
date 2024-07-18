from src.model import single_node, single_node_xavier, HGNN_ATT
import timeit
from itertools import chain
import torch
from src.timer import Timer
from src.loss import loss_cal_and_update, maxcut_loss_func_helper, loss_maxcut_weighted, loss_sat_weighted, loss_maxind_weighted, loss_maxind_QUBO, loss_maxind_weighted2, loss_task_weighted, loss_maxcut_weighted_anealed, loss_task_weighted_vec, loss_mincut_weighted, loss_partitioning_weighted, loss_partitioning_nonbinary, loss_maxcut_weighted_coarse, loss_maxind_QUBO_coarse, loss_maxcut_weighted_multi
from src.utils import mapping_algo, mapping_distribution, gen_q_mis,gen_q_maxcut, mapping_distribution_QUBO, get_normalized_G_from_con, mapping_distribution_vec_task, mapping_distribution_vec, all_to_weights, all_to_weights_task
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from torch.autograd import grad
import pickle
import time

def centralized_train(G, params, f, C, n, info, file_name):
    temp_time = timeit.default_timer()
    ####### fix seed to ensure consistent results ######
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG

    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    #### sometimes we want the number of epochs to grow with n #####
    # rounds = max(int(2 * n // 10), int(params['epoch']))
    rounds = int(params['epoch'])
    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]

    ### q_torch helps compute graph MIS and Maxcut loss faster
    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    elif params['mode'] == 'QUBO_maxcut':
        q_torch = gen_q_maxcut(C, n, torch_dtype=None, torch_device=None)

    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}
    if params['mode']=='partition':
        outrange=params['n_partitions']
        outbias=0
    else:
        outrange=1
        outbias=0


    ##### transfer learning: load and freeze the layers and only optimize on the input embeding ######
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

    ###### if we want to initialize our model and input embedding by a pretrained model ####
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())

    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    #inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))

    #### computes the distance between node features at each layer to detect oversmoothing ####
    dist=[]

    for i in range(rounds):
        ##### forward path ######
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        # dis1=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        # dis2=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.relu(temp)
        temp = conv2(temp)
        # dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        # dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.sigmoid(temp)
        temp = temp * outrange + outbias
        #temp = torch.softmax(temp, dim=0)
        # dist.append([dis1,dis2,dis3,dis4])

        ###### compute the loss #######
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])

        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], params['hyper'])

        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])

        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)

        elif params['mode'] == 'QUBO_maxcut':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
            # print(loss,loss2)

        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])

        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break

        elif params['mode'] == 'mincut':
            loss = loss_mincut_weighted(temp, C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'],
                                        indicest, params['hyper'])


        elif params['mode'] == 'partition':
            loss = loss_partitioning_nonbinary(temp, C, params['n_partitions'], [1 for i in range(len(C))], params['hyper'])
        #     loss = loss_partitioning_weighted(temp, C, weights, params['hyper'])
        # ###### optimization step #######
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()


        ##### decide if we want to stop based on tolerance (params['tol']) and patience (params['patience']) ######
        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')

                #### save the model #####
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0

        #### keep the best loss and result ####3
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')

            ##### the end of the epochs #####
            if i==int(params['epoch'])-1:
                ##### save the model #####
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
                ##### save the model #####
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss=loss

   

    if params['load_best_out']:
        with open('best_out.txt', 'r') as f:
            best_out=eval(f.read())
    else:
        best_out = best_out.detach().numpy()
        best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    all_weights = [1.0 for c in (C)]
    if params['data'] != 'task':
        weights = all_to_weights(all_weights, n, C)
    else:
        weights = all_to_weights_task(all_weights, n, C)

    #### plot the histogram of the HyperGNN output (see if it's learning anything) #####
    name = params['plot_path']+ file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()
    train_time = timeit.default_timer() - temp_time


    ##### fine-tuning ######
    temp_time2 = timeit.default_timer()
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


##### for multi-gpu (distributed) training #####
def centralized_train_for(params, f, total_C, n, info_input_total, weights, file_name, device=0,
                          inner_constraint=None, outer_constraint=None, cur_nodes=None, inner_info=None,
                          outer_info=None):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cuda:' + str(device))
    TORCH_DTYPE = torch.float32
    verbose = False

    if inner_constraint is not None:
        total_C = total_C
        C = inner_constraint
        info_input_total = info_input_total
        info_input = inner_info

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p = 0
    count = 0
    prev_loss = 100
    patience = params['patience']
    best_loss = float('inf')
    dct = {x + 1: x for x in range(len(weights))}
    

    print("[n]", n, "[C]", len(C), "weight", len(weights))
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(f, f // 2).to(TORCH_DEVICE)
        conv2 = single_node_xavier(f // 2, 1).to(TORCH_DEVICE)
        # conv2 = single_node(f//2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        conv1 = conv1.to(TORCH_DEVICE)
        conv2 = conv2.to(TORCH_DEVICE)
        embed = embed.to(TORCH_DEVICE)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())

    # set up multi gpu
    if params["multi_gpu"]:
        if TORCH_DEVICE == torch.device("cpu"):
            conv1 = torch.nn.parallel.DistributedDataParallel(conv1, device_ids=None, output_device=None)
            conv2 = torch.nn.parallel.DistributedDataParallel(conv2, device_ids=None, output_device=None)
        else:
            conv1 = torch.nn.parallel.DistributedDataParallel(conv1, device_ids=[TORCH_DEVICE],
                                                              output_device=TORCH_DEVICE)
            conv2 = torch.nn.parallel.DistributedDataParallel(conv2, device_ids=[TORCH_DEVICE],
                                                              output_device=TORCH_DEVICE)

    if params["multi_gpu"]:
        optimizer = torch.optim.Adam(parameters, lr=params['lr'])
        inputs = embed.weight.to(TORCH_DEVICE)
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(info_input)
    else:
        optimizer = torch.optim.Adam(parameters, lr=params['lr'])
        inputs = embed.weight.to(TORCH_DEVICE)
    if params["test_multi_gpu"] and not params["multi_gpu"]:
        # random select n//4 from info
        selected_indx = random.sample(range(1, n + 1), n // 4)
        info = {i + 1: info_input[selected_indx[i]] for i in range(len(selected_indx))}
        con_list_length = len(selected_indx)
    elif params["multi_gpu"]:
        info = info_input
        con_list_range_keys = list(info.keys())
        con_list_range = [i for i in con_list_range_keys if len(info[i]) > 0]
        # print("con_list_range", con_list_range)
    else:
        con_list_length = n
        info = info_input

    start = con_list_range_keys[0]  # start = 1, 501, 1001, 1501
    for ep in range(int(params['epoch'])):
        temp = conv1(inputs)
        temp2 = torch.ones(temp.shape).to(TORCH_DEVICE)
        st_start = time.time()
        st = time.time()

        for i in con_list_range:
            cons_list = info[i]
            indices = [cons[1] if cons[0] == i else cons[0] for cons in cons_list]
            indices_tensor = torch.tensor(indices, dtype=torch.long)  # Convert list to long tensor for indexing
            indices_tensor = indices_tensor - start
            idx = i - start
            temp2[idx, :] += torch.sum(temp[indices_tensor, :], dim=0).to(TORCH_DEVICE)
            temp2[idx, :] /= len(info[i])

        temp = temp2
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp2 = torch.ones(temp.shape).to(TORCH_DEVICE)

        for i in con_list_range:
            cons_list = info[i]
            indices = [cons[1] if cons[0] == i else cons[0] for cons in cons_list]
            indices_tensor = torch.tensor(indices, dtype=torch.long)  # Convert list to long tensor for indexing
            indices_tensor = indices_tensor - start
            idx = i - start
            temp2[idx, :] += torch.sum(temp[indices_tensor, :], dim=0).to(TORCH_DEVICE)
            temp2[idx, :] /= len(info[i])
        temp = temp2
        temp = torch.sigmoid(temp)
        et = time.time()
        if verbose:
            print("Prepare data to compute loss: ", et - st)

        st = time.time()
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            if params["multi_gpu"]:
                temp_reduce = [torch.zeros_like(temp).to(f'cuda:{device}') for _ in range(4)]
                torch.distributed.all_gather(temp_reduce, temp)
                temp_reduce = torch.cat(temp_reduce, dim=0)
                temp_reduce = temp_reduce.squeeze(1)
            loss = loss_maxcut_weighted_multi(temp, C, dct, torch.ones(len(C) + len(outer_constraint)).to(TORCH_DEVICE),
                                        params['hyper'],
                                        TORCH_DEVICE, outer_constraint, temp_reduce, start=start)
        elif params['mode'] == 'maxind':
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        et = time.time()
        if verbose:
            print("Compute forward loss for maxcut: ", et - st)

        st = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        et = time.time()
        if verbose:
            print("Backward loss: ", et - st)

        st = time.time()
        if params["multi_gpu"]:
            loss_sum = loss.clone()
            torch.distributed.reduce(loss_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
            if torch.distributed.get_rank() == 0:
                average_loss = loss_sum / torch.distributed.get_world_size()
                if average_loss < best_loss:
                    print("average_loss", average_loss, "best_loss", best_loss)
                    best_loss = average_loss
                    best_out = temp_reduce.cpu()
                    # print("best_out", best_out)
                    if i == int(params['epoch']) - 1:
                        name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                        torch.save(embed, name)
                        name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                        torch.save(conv1, name)
                        name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                        torch.save(conv2, name)
        else:
            if loss < best_loss:
                best_loss = loss
                best_out = temp
                print(f'found better loss')
                if i == int(params['epoch']) - 1:
                    name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                    torch.save(embed, name)
                    name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                    torch.save(conv1, name)
                    name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                    torch.save(conv2, name)
        prev_loss = loss
        et = time.time()
        if verbose:
            print("Update best loss: ", et - st)
        print("Epoch", ep, "Epoch time: ", et - st_start, "current loss", loss)

    if params["multi_gpu"]:
        if torch.distributed.get_rank() == 0:
            best_out = best_out.detach().numpy()
            print("best_out", best_out)
            best_out = {i + 1: best_out[i] for i in range(len(best_out))}
            with open("best_out.txt", "w") as f:
                f.write(str(best_out))
            train_time = timeit.default_timer() - temp_time
            temp_time2 = timeit.default_timer()
            all_weights = [1.0 for c in (total_C)]
            print("info_input_total", len(info_input_total), "weights", len(weights), "total_C", len(total_C))
            res = mapping_distribution(best_out, params, len(weights), info_input_total, weights, total_C, all_weights,
                                       1, params['penalty'], params['hyper'])
            print("res", res)
            map_time = timeit.default_timer() - temp_time2
        else:
            res = None
            train_time = None
            map_time = None
            best_out = None
    else:
        best_out = best_out.detach().numpy()
        best_out = {i + 1: best_out[i][0] for i in range(len(best_out))}
        train_time = timeit.default_timer() - temp_time
        temp_time2 = timeit.default_timer()
        all_weights = [1.0 for c in (C)]
        res = mapping_distribution(best_out, params, n, info_input, weights, C, all_weights, 1, params['penalty'],
                                   params['hyper'])
        print("res", res)
        map_time = timeit.default_timer() - temp_time2
    return res, best_out, train_time, map_time


##### gradient descent solver (no HyperGNN) ####
def GD_train(params, f, C, n, info, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32
    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}


    embed = nn.Embedding(n, 1)

    parameters = embed.parameters()
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    for i in range(int(params['epoch'])):
        print(i)
        inputs = embed.weight
        temp = torch.sigmoid(inputs)
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C,  [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'],indicest, params['hyper'])
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
    if params['data'] != 'task':
        weights = all_to_weights(all_weights, n, C)
    else:
        weights = all_to_weights_task(all_weights, n, C)

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

def centralized_train_vec(G, params, C, n, info, file_name, L):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32


    # if params['mode'] == 'task_vec':
    #     L=len(C)
    # else:
    #     L=params['n_partitions']
    #f = n // 4
    if params['f_input']:
        f = params['f']
    else:
        f=10
    if params['mode'] == 'task_vec':
        C_dic = {}
        ic = 0
        lenc = torch.zeros([L])
        for c in C:
            lenc[ic] = len(c)
            C_dic[str(c)] = ic
            ic += 1
        leninfo = torch.zeros([n])
        for inn in range(n):
            leninfo[inn] = len(info[inn + 1])
    # C_mat=np.zeros([n,L])
    # for c in C_dic.keys():
    #     for i in c:
    #         C_mat[i-1, C_dic[str(c)]]=1

    temper0 = 0.01
    p = 0
    count = 0
    prev_loss = 100
    patience = params['patience']
    best_loss = float('inf')
    dct = {x + 1: x for x in range(n)}

    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        # embed = nn.Embedding(n, f)
        # embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # # for param in embed.parameters():
        #     param.requires_grad = False
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, L*f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(L*f, L*f // 2)
        conv2 = single_node_xavier(L*f // 2, L)
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
    optimizer = torch.optim.Adam(parameters, lr=params['lr'])
    # inputs = embed.weight
    # grad1=torch.zeros((int(params['epoch'])))
    # grad2 = torch.zeros((int(params['epoch'])))
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        temp = G @ temp
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp = G @ temp
        # temp = torch.sigmoid(temp)
        temp = torch.softmax(temp, dim=1)
        if params['mode'] == 'task_vec':
            loss = loss_task_weighted_vec(temp, lenc, leninfo)
            if loss == 0:
                print("found zero loss")
                break
        elif params['mode']== 'partition':
            loss = loss_partitioning_weighted(temp, C, [1 for i in range(len(C))], params['hyper'])
        elif params['mode']== 'MNP':
            loss = loss_MNP_weighted(temp, C, [1 for i in range(len(C))], params['hyper'])


        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i == int(params['epoch']) - 1:
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
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
        prev_loss = loss

    best_out = best_out.detach().numpy()
    best_out_d = {i + 1: best_out[i,:] for i in range(len(best_out))}
    train_time = timeit.default_timer() - temp_time
    temp_time2 = timeit.default_timer()

    all_weights = [1.0 for c in (C)]
    if params['data'] != 'task':
        weights = all_to_weights(all_weights, n, C)
    else:
        weights = all_to_weights_task(all_weights, n, C)

    #name = params['plot_path'] + file_name[:-4] + '.png'
    # plt.hist(best_out_d.values(), bins=np.linspace(0, 1, 50))
    # plt.savefig(name)
    # plt.show()

    if params['mode']=='task_vec':
        leninfon=torch.Tensor.numpy(leninfo)
        lencn=torch.Tensor.numpy(lenc)
        best_res = mapping_distribution_vec_task(best_out_d, params, n, info,  C, C_dic, all_weights, 1, lencn,leninfon,params['penalty'],
                                       params['hyper'])

    elif params['mode']=='partition':
        best_res = mapping_distribution_vec(best_out_d, params, n, info, weights, C, all_weights, 1, L, params['penalty'],
                                                 params['hyper'])
    map_time = timeit.default_timer() - temp_time2

    return best_res, best_out, train_time, map_time







def centralized_train_att( H, params, f, C, n, info, file_name):
    temp_time = timeit.default_timer()

    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG

    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]


    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}


    ### have not fixed the transfer learning for ATT ####
    if params['transfer']:
        name = params["model_load_path"] + 'ATT.pt'
        model_att = torch.load(name)
        for param in model_att.parameters():
            param.requires_grad = False

    #### define the HyperGAT model ####
    else:
        model_att=HGNN_ATT(n, f, 3*f//4, 1, params)


    if params["initial_transfer"]:
        name = params["model_load_path"] + 'ATT.pt'
        model_att = torch.load(name)


    dist=[]
    for i in range(int(params['epoch'])):
        print(i)

        #### forward path ####
        temp=model_att(torch.Tensor(H).float())


        ##### calculate the loss #####
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])

        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C,  [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], indicest, params['hyper'])

        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])

        # elif params['mode'] == 'QUBO':
        #     # probs = temp[:, 0]
        #     loss = loss_maxind_QUBO(temp, q_torch)
        #     # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
        #     # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        #     # print(loss, loss3)
        # elif params['mode'] == 'QUBO_maxcut':
        #     # probs = temp[:, 0]
        #     loss = loss_maxind_QUBO(temp, q_torch)
        #     #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
        #     # print(loss,loss2)

        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])

        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break

        ##### optimization step ####
        model_att.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        model_att.optimizer.step()

        ##### decide if we want to stop based on tolerance (params['tol']) and patience (params['patience']) ######
        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')

                ### if we want to save the model ###
                # name = params["model_save_path"] + 'ATT.pt'
                # torch.save(model_att, name)
                break
        else:
            count = 0

        #### keep the best loss and result ####
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            # if i==int(params['epoch'])-1:
                ### if we want to save the model ###
                # name = params["model_save_path"] + 'ATT.pt'
                # torch.save(model_att, name)

        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                ### if we want to save the model ###
                # name = params["model_save_path"] + 'ATT.pt'
                # torch.save(model_att, name)
                break
        prev_loss=loss

    


    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer() - temp_time

    all_weights = [1.0 for c in (C)]
    if params['data'] != 'task':
        weights = all_to_weights(all_weights, n, C)
    else:
        weights = all_to_weights_task(all_weights, n, C)

    #### plot the histogram of the HyperGNN output to see if it's learning anything ####
    name = params['plot_path']+ file_name[:-4] + '.png'
    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.savefig(name)
    plt.show()

    ##### fine-tuning ####
    temp_time2 = timeit.default_timer()
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



###### bipartite GNN #####
def centralized_train_bipartite( G, params, f, C, n, n_hyper, info,  file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    # if params['mode'] == 'QUBO':
    #     q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    # elif params['mode'] == 'QUBO_maxcut':
    #     q_torch = gen_q_maxcut(C, n, torch_dtype=None, torch_device=None)
    if params['hyper']:
        indicest = [[i - 1 for i in c] for c in C]
    else:
        indicest = [[i - 1 for i in c[0:2]] for c in C]

    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')

    if params['transfer']: #not updated for bipartite
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

        #4 layers
        # conv1 = single_node_xavier(f, f)
        # conv2 = single_node_xavier(f, f // 2)
        # conv3 = single_node_xavier(f // 2, f // 2)
        # conv4 = single_node_xavier(f // 2, 1)
        # parameters = chain(conv1.parameters(), conv2.parameters(), conv3.parameters(), conv4.parameters(), embed.parameters())

        #two layers
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(),embed.parameters())

    if params["initial_transfer"]: #not updated for bipartite
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    #inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))
    dist=[]
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        # dis1=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        # dis2=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.relu(temp)
        temp = conv2(temp)
        # dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp


        # 4 layers
        # temp = torch.relu(temp)
        # temp = conv3(temp)
        # temp = G @ temp
        # temp = torch.relu(temp)
        # temp = conv4(temp)
        # temp = G @ temp



        # dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.sigmoid(temp)
        #temp = torch.softmax(temp, dim=0)
        # dist.append([dis1,dis2,dis3,dis4])
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp[0:n_hyper], C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], indicest, params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)
        elif params['mode'] == 'QUBO_maxcut':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
            # print(loss,loss2)
        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])
        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
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
    best_out = {i+1: best_out[i][0] for i in range(n_hyper)}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    if params['data'] != 'task':
        weights = all_to_weights(all_weights, n, C)
    else:
        weights = all_to_weights_task(all_weights, n, C)

    # name = params['plot_path']+ file_name[:-4] + '.png'
    # plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    # plt.savefig(name)
    # plt.show()

    res = mapping_distribution(best_out, params, n_hyper, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2

    return res, best_out, train_time, map_time



def centralized_train_cliquegraph(G, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    # seed_value = 100
    # random.seed(seed_value)  # seed python RNG
    # np.random.seed(seed_value)  # seed global NumPy RNG
    # torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    # if params['mode'] == 'QUBO':
    #     q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    # elif params['mode'] == 'QUBO_maxcut':
    #     q_torch = gen_q_maxcut(C, n, torch_dtype=None, torch_device=None)


    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')

    if params['transfer']: #not updated for bipartite
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

        #4 layers
        # conv1 = single_node_xavier(f, f)
        # conv2 = single_node_xavier(f, f // 2)
        # conv3 = single_node_xavier(f // 2, f // 2)
        # conv4 = single_node_xavier(f // 2, 1)
        # parameters = chain(conv1.parameters(), conv2.parameters(), conv3.parameters(), conv4.parameters(), embed.parameters())

        #two layers
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(),embed.parameters())

    if params["initial_transfer"]: #not updated for bipartite
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    #inputs = embed.weight
    #grad1=torch.zeros((int(params['epoch'])))
    #grad2 = torch.zeros((int(params['epoch'])))
    dist=[]
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        # dis1=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        # dis2=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.relu(temp)
        temp = conv2(temp)
        # dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp


        # 4 layers
        # temp = torch.relu(temp)
        # temp = conv3(temp)
        # temp = G @ temp
        # temp = torch.relu(temp)
        # temp = conv4(temp)
        # temp = G @ temp



        # dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.sigmoid(temp)
        #temp = torch.softmax(temp, dim=0)
        # dist.append([dis1,dis2,dis3,dis4])
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, [1 for i in range(len(C))], params['penalty_inc'], params['penalty_c'], params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            # loss2 = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            # loss3 = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
            # print(loss, loss3)
        elif params['mode'] == 'QUBO_maxcut':
            # probs = temp[:, 0]
            loss = loss_maxind_QUBO(temp, q_torch)
            #loss2 = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
            # print(loss,loss2)
        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])
        elif params['mode'] == 'task':
            loss = loss_task_weighted(temp, C, dct, [1 for i in range(len(C))])
            if loss==0:
                print("found zero loss")
                break
        optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
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
    best_out = {i+1: best_out[i][0] for i in range(n)}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    name = params['plot_path'] + file_name[:-4] + '.png'
    # plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    # plt.savefig(name)
    # plt.show()
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2

    return res, best_out, train_time, map_time



def centralized_train_coarsen(G, params, f, C, org_constraints, graph_dict, n_org, n, info, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32

    if params["mode"] == "QUBO":
        q_torch = gen_q_mis(org_constraints, n_org, 2, torch_dtype=None, torch_device=None)
    elif params["mode"] == "QUBO_maxcut":
        q_torch = gen_q_maxcut(org_constraints, n_org, torch_dtype=None, torch_device=None)

    temper0 = 0.01
    p = 0
    count = 0
    prev_loss = 100000
    patience = params["patience"]
    best_loss = float("inf")
    dct = {x + 1: x for x in range(n)}

    if params["transfer"]:
        name = params["model_load_path"] + "embed_" + file_name[:-4] + ".pt"
        embed = torch.load(name)
        name = params["model_load_path"] + "conv1_" + file_name[:-4] + ".pt"
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"] + "conv2_" + file_name[:-4] + ".pt"
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 2)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + "conv1_" + file_name[:-4] + ".pt"
        conv1 = torch.load(name)
        name = params["model_load_path"] + "conv2_" + file_name[:-4] + ".pt"
        conv2 = torch.load(name)
        name = params["model_load_path"] + "embed_" + file_name[:-4] + ".pt"
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr=params["lr"])

    dist = []
    for i in range(int(params["epoch"])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        dis1 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(
            np.linalg.norm(temp.detach().numpy(), axis=1)
        )
        temp = G @ temp
        dis2 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(
            np.linalg.norm(temp.detach().numpy(), axis=1)
        )
        temp = torch.relu(temp)
        temp = conv2(temp)
        dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(
            np.linalg.norm(temp.detach().numpy(), axis=1)
        )
        temp = G @ temp
        dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(
            np.linalg.norm(temp.detach().numpy(), axis=1)
        )
        temp = torch.sigmoid(temp)
        dist.append([dis1, dis2, dis3, dis4])
        if params["mode"] == "sat":
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params["mode"] == "maxcut":
            #temp = compute_prob_org(graph_dict, temp, n_org)
            #dct = {x + 1: x for x in range(n_org)}
            loss = loss_maxcut_weighted_coarse(
                temp, org_constraints, graph_dict, [1 for i in range(len(org_constraints))], params["hyper"]
            )
        elif params["mode"] == "maxind":
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params["mode"] == "QUBO":
            loss = loss_maxind_QUBO_coarse(temp, q_torch, graph_dict, n_org)


        elif params["mode"] == "QUBO_maxcut":
            loss = loss_maxind_QUBO(temp, q_torch)
        elif params["mode"] == "maxcut_annea":
            temper = temper0 / (1 + i)
            loss = loss_maxcut_weighted_anealed(
                temp, C, dct, [1 for i in range(len(C))], temper, params["hyper"]
            )
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params["tol"]) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params["patience"]:
                print(f"Stopping early on epoch {i} (patience: {patience})")
                name = params["model_save_path"] + "embed_" + file_name[:-4] + ".pt"
                torch.save(embed, name)
                name = params["model_save_path"] + "conv1_" + file_name[:-4] + ".pt"
                torch.save(conv1, name)
                name = params["model_save_path"] + "conv2_" + file_name[:-4] + ".pt"
                torch.save(conv2, name)
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f"found better loss")
            if i == int(params["epoch"]) - 1:
                name = params["model_save_path"] + "embed_" + file_name[:-4] + ".pt"
                torch.save(embed, name)
                name = params["model_save_path"] + "conv1_" + file_name[:-4] + ".pt"
                torch.save(conv1, name)
                name = params["model_save_path"] + "conv2_" + file_name[:-4] + ".pt"
                torch.save(conv2, name)
        else:
            p += 1
            if p > params["patience"]:
                print("Early Stopping")
                name = params["model_save_path"] + "embed_" + file_name[:-4] + ".pt"
                torch.save(embed, name)
                name = params["model_save_path"] + "conv1_" + file_name[:-4] + ".pt"
                torch.save(conv1, name)
                name = params["model_save_path"] + "conv2_" + file_name[:-4] + ".pt"
                torch.save(conv2, name)
                break
        prev_loss = loss

    

    if params["load best out"]:
        with open("best_out.txt", "r") as f:
            best_out = eval(f.read())
    else:
        best_out = best_out.detach().numpy()
        #best_out = {i + 1: best_out[i][0] for i in range(len(best_out))}
        best_out = {i + 1: best_out[graph_dict[i+1][0]-1][graph_dict[i+1][1]] for i in range(n_org)}

    train_time = timeit.default_timer() - temp_time
    temp_time2 = timeit.default_timer()
    all_weights = [1.0 for c in range(len(org_constraints))]
    weights = all_to_weights_task(all_weights, n_org, org_constraints)

    plt.hist(best_out.values(), bins=np.linspace(0, 1, 50))
    plt.show()

    if params["mapping"] == "distribution":
        res = mapping_distribution(
            best_out,
            params,
            n_org,
            info,
            weights,
            org_constraints,
            all_weights,
            1,
            params["penalty"],
            params["hyper"],
        )
    elif params["mapping"] == "threshold":
        res = {x: 0 if best_out[x] < 0.5 else 1 for x in best_out.keys()}

    map_time = timeit.default_timer() - temp_time2
    return res, best_out, train_time, map_time
