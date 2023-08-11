from src.utils import generate_H_from_edges, _generate_G_from_H, generate_H_from_constraints, all_to_weights, gen_q_mis, get_normalized_G_from_con, Maxind_postprocessing
import numpy as np
import torch
from src.timer import Timer
from src.sampler import Sampler
import timeit
from src.trainer import Trainer, centralized_train, GD_train
from src.loss import loss_maxcut_numpy_boost, loss_sat_numpy_boost, loss_maxind_numpy_boost, loss_maxind_QUBO
import matplotlib.pyplot as plt
import dgl

import random
import os
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from dgl.nn.pytorch import GraphConv
from itertools import chain, islice, combinations
from networkx.algorithms import maximal_independent_set as mis
from time import time

#class solver:
#    def __init__(constraints, header, params, log) -> None:
        

def local_solver(constraints, header, params, log):
    n = header['num_nodes']
    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)

    #f = int(np.sqrt(n))
    f = n // 2
    info = {x+1:[] for x in range(n)}
    for constraint in constraints:
        for node in constraint:
            info[abs(node)].append(constraint) 
    if params['fast']:
        contraints_0 = np.array(constraints) - 1 
        H = generate_H_from_edges(contraints_0, n)
        G = _generate_G_from_H(H)
        D = np.diagonal(G).copy()
        np.fill_diagonal(G, 0)
        G = torch.from_numpy(G).float()
        D = torch.from_numpy(D).float()
    else:
        Gs = {}
        dcts = {}
        for node in info:
            H, dct = generate_H_from_constraints(info[node], node)
            G = _generate_G_from_H(H)[0]
            Gs[node] = torch.from_numpy(G).float()
            dcts[node] = dct
    # sampler initialization
    sampler = Sampler(params, n, f)
    # timer initialization
    time = Timer()               
    all_weights = [[1.0 for c in (constraints)] for i in range(params['num_samples'])]
    weights = [all_to_weights(all_weights[i], n, constraints) for i in range(len(all_weights))]
    reses = []
    temp_time = timeit.default_timer()
    for i in range(params['K']):
        #print(weights)
        scores = []
        Xs = sampler.get_Xs(i)
        temp_weights = []
        for j in range(params['num_samples']):
            temp_trainer = Trainer(params, f, n, info, log, weights[j], time)
            temp_trainer.train(dcts, Xs[j], Gs)
            res = temp_trainer.map(constraints, all_weights[j], inc=params['boosting_mapping'])
            if params['mode'] == 'sat':
                score, new_w = loss_sat_numpy_boost(res, constraints, all_weights[j], inc=params['boosting_mapping'])
                scores.append(score)
            elif params['mode'] == 'maxcut':
                score, new_w = loss_maxcut_numpy_boost(res, constraints, all_weights[j], inc=params['boosting_mapping'])
                scores.append(score)
            elif params['mode'] == 'maxind':
                score, score1, new_w = loss_maxind_numpy_boost(res, constraints, all_weights[j], inc=params['boosting_mapping'])
                scores.append(score)
            elif params['mode'] == 'QUBO':
                score = loss_maxind_QUBO(res, q_torch)
                scores.append(score)
                new_w=weights
            temp_weights.append(new_w)
        weights = [all_to_weights(temp_weights[i], n, constraints) for i in range(len(temp_weights))]
        sampler.update(scores)
        reses.append(scores)
    time.total_time += (timeit.default_timer() - temp_time)
    return reses, time.__dict__           


def centralized_solver(constraints, header, params, file_name):
    temp_time = timeit.default_timer()
    edges = [[abs(x) - 1 for x in edge] for edge in constraints]
    n = header['num_nodes']
    #if params['mode'] == 'QUBO':
    q_torch = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)
    f = int(np.sqrt(n))
    #f=n // 2
    Gn = get_normalized_G_from_con(constraints, header)
    H = generate_H_from_edges(edges, n)
    G = _generate_G_from_H(H)
    G = torch.from_numpy(G).float()
    info = {x+1:[] for x in range(n)}
    for constraint in constraints:
        for node in constraint:
            info[abs(node)].append(constraint) 
    all_weights = [[1.0 for c in (constraints)] for i in range(params['num_samples'])]
    weights = [all_to_weights(all_weights[i], n, constraints) for i in range(len(all_weights))]
    # sampler initialization
    sampler = Sampler(params, n, f)
    # timer initialization            
    reses = []
    reses2 = []
    reses_th = []
    probs = []
    for i in range(params['K']):
        #print(weights)
        scores = []
        scores2 = []
        scores_th = []
        scores1 = []
        Xs = sampler.get_Xs(i)
        temp_weights = []
        for j in range(params['num_samples']):
            #res, res2, prob = centralized_train(Xs[j], Gn, params, f, constraints, n, info, weights[i])
            if not params["GD"]:
                res,  prob , train_time, map_time= centralized_train(Xs[j], Gn, params, f, constraints, n, info, weights[i], file_name)
            else:
                res, prob, train_time, map_time = GD_train(Xs[j], G, params, f, constraints, n, info, weights[i], file_name)

            res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}
            if params['mode'] == 'sat':
                score, new_w = loss_sat_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                scores.append(score)
            elif params['mode'] == 'maxcut':
                score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                score_th, _ =  loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                scores.append(score)
                scores_th.append(score_th)
            elif params['mode'] == 'maxind':
                res_feas=Maxind_postprocessing(res,constraints, n)
                score, score1, new_w = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                score_th, score1, new_w = loss_maxind_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],inc=params['boosting_mapping'])
                print(score, score1)
                scores.append(score)
                scores1.append(score1)
                scores_th.append(score_th)
                #score2, score12, new_w2 = loss_maxind_numpy_boost(res2, constraints, [1 for i in range(len(constraints))],inc=params['boosting_mapping'])
                #scores2.append(score2)
            elif params['mode'] == 'QUBO':
                res_feas = Maxind_postprocessing(res, constraints, n)
                res_th_feas = Maxind_postprocessing(res_th, constraints, n)
                score = loss_maxind_QUBO(torch.Tensor(list(res_feas.values())), q_torch)
                # score_old, score1, _ = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                #                                                inc=params['boosting_mapping'])
                score_th = loss_maxind_QUBO(torch.Tensor(list(res_th_feas.values())), q_torch)
                scores.append(score)
                scores_th.append(score_th)
                #scores1.append(score1)
                #print(score1)
                # print(score_old)
                #score2 = loss_maxind_QUBO(torch.Tensor(list(res2.values())), q_torch)
                #scores2.append(score2)
            # score, _ = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
            # scores.append(score)
            probs.append(prob)
        sampler.update(scores)
        reses.append(scores)
        #reses2.append(scores2)
        reses_th.append(scores_th)
    return reses, reses2, reses_th, probs, timeit.default_timer() - temp_time, train_time, map_time

from src.data_reading import read_uf, read_stanford, read_hypergraph
import numpy as np
import torch
import random
def gradient_solver(constraints, header, params):
    def loss_maxcut(probs, C):
        loss = 0
        for c in C:
            temp_1s = 1
            temp_0s = 1
            for index in c:
                temp_1s *= (1 - probs[index-1])
                temp_0s *= (probs[index-1])
            temp = (temp_1s + temp_0s - 1)
            loss += temp
        return loss
    def loss_maxind(probs, C):
        loss = -sum(probs)
        for c in C:
            temp = 2 * probs[c[0]-1] * probs[c[1]-1]
            loss += (temp)
        return loss
    def mapping(best_outs, params, n, info):
        for rea in range(params['N_realize']):
            res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
            ord = random.sample(range(0, n), n)
            t = 1
            for it in range(params['Niter_h']):
                res1 = res
                for i in ord:
                    temp = res.copy()
                    # temp = pr.copy()
                    if res[i] == 0:
                        temp[i] = 1
                    else:
                        temp[i] = 0
                    lt= loss_(temp, info[i+1])
                    l1= loss_(res, info[i+1])
                    if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                        res = temp
                t = t * 0.95
        return res

    n = header['num_nodes']
    info = {x+1:[] for x in range(n)}
    for constraint in constraints:
        for node in constraint:
            info[abs(node)].append(constraint) 
    temp_time = timeit.default_timer()
    probs = torch.rand(n, requires_grad=True)
    if params['mode'] == 'maxcut':
        loss_ = loss_maxcut
    elif params['mode'] == 'maxind':
        loss_ = loss_maxind
    last_loss = 0
    for i in range(int(params['epoch'])):
        probs_ = torch.sigmoid(probs)
        loss = loss_(probs_, constraints)
        loss.backward()
        if (last_loss - loss).item() <= 1e-2:
            last_loss = loss
            break
        if i % 100 == 0:
            print(f'at epoch {i}, loss = {loss.item()}')
        probs = probs.clone().detach() - probs.grad * params['lr']
        probs.requires_grad_()
    probs = torch.sigmoid(probs)
    probs = probs.detach().numpy()
    probs = {i: probs[i] for i in range(len(probs))}
    plt.hist(probs.values(), bins=np.linspace(0, 1, 50), weights=1 / 2000 * np.ones([n, ]))
    res = mapping(probs, params, n, info)
    reses = loss_(res, constraints)
    return reses, timeit.default_timer() - temp_time


from src.QUBO_utils import generate_graph, get_gnn, run_gnn_training, qubo_dict_to_torch, gen_combinations, loss_func
def QUBO_solver(params):
    # fix seed to ensure consistent results
    seed_value = 1
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG

    # Set GPU/CPU
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DTYPE = torch.float32
    print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


    # MacOS can have issues with MKL. For more details, see
    # https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    def gen_q_dict_mis(nx_G, penalty=2):
        """
        Helper function to generate QUBO matrix for MIS as minimization problem.

        Input:
            nx_G: graph as networkx graph object (assumed to be unweigthed)
        Output:
            Q_dic: QUBO as defaultdict
        """

        # Initialize our Q matrix
        Q_dic = defaultdict(int)

        # Update Q matrix for every edge in the graph
        # all off-diagonal terms get penalty
        for (u, v) in nx_G.edges:
            Q_dic[(u, v)] = penalty

        # all diagonal terms get -1
        for u in nx_G.nodes:
            Q_dic[(u, u)] = -1

        return Q_dic





    # Calculate results given bitstring and graph definition, includes check for violations
    def postprocess_gnn_mis(best_bitstring, nx_graph):
        """
        helper function to postprocess MIS results

        Input:
            best_bitstring: bitstring as torch tensor
        Output:
            size_mis: Size of MIS (int)
            ind_set: MIS (list of integers)
            number_violations: number of violations of ind.set condition
        """

        # get bitstring as list
        bitstring_list = list(best_bitstring)

        # compute cost
        size_mis = sum(bitstring_list)

        # get independent set
        ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
        edge_set = set(list(nx_graph.edges))

        print('Calculating violations...')
        # check for violations
        number_violations = 0
        for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
            number_violations += len(set(ind_set_chunk).intersection(edge_set))

        return size_mis, ind_set, number_violations

    # Graph hypers
    n = 100
    d = 3
    p = 2
    graph_type = 'reg'

    # NN learning hypers #
    number_epochs = int(1e5)
    learning_rate = 1e-4
    PROB_THRESHOLD = 0.5

    # Early stopping to allow NN to train to near-completion
    tol = 1e-4  # loss must change by more than tol, or trigger
    patience = 100  # number early stopping triggers before breaking loop

    # Problem size (e.g. graph size)
    n = 100

    # Establish dim_embedding and hidden_dim values
    dim_embedding = int(np.sqrt(n))  # e.g. 10
    hidden_dim = int(dim_embedding / 2)  # e.g. 5

    # Constructs a random d-regular or p-probabilistic graph
    nx_graph = generate_graph(n=n, d=d, p=p, graph_type=graph_type, random_seed=seed_value)
    # get DGL graph from networkx graph, load onto device
    graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)
    q_torch = qubo_dict_to_torch(nx_graph, gen_q_dict_mis(nx_graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

    #folder_path= "./data/maxind_data/random_regular/reg_graph_100_single/"
    folder_path =params["folder_path"]
    for file in os.listdir(folder_path):
    #     path = folder_path + file_name
    #if True:
        if file.startswith('G'):
            path=folder_path+file
            constraints, header = read_stanford(path)


            edges = [[abs(x) - 1 for x in edge] for edge in constraints]


            # q_torch_1 = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)

            # f = int(np.sqrt(n))
            # H = generate_H_from_edges(edges, n)
            # G = _generate_G_from_H(H)

            # nx_graph2 = dgl.graph(edges)
            # graph_dgl2 = dgl.from_networkx(nx_graph=nx_graph2)
            # graph_dgl2 = graph_dgl2.to(TORCH_DEVICE)
            nx_graph2 = nx.Graph()
            nx_graph2.add_edges_from(edges)
            nodes_l=list(nx_graph2.nodes)
            nodes_l.sort()
            nodes_d={nodes_l[i]: i for i in range(len(nodes_l)) }
            edges_s=[[nodes_d[x] for x in edge] for edge in edges]
            nx_graph2 = nx.Graph()
            nx_graph2.add_edges_from(edges_s)
            graph_dgl2 = dgl.from_networkx(nx_graph=nx_graph2)
            graph_dgl2 = graph_dgl2.to(TORCH_DEVICE)

            # Construct Q matrix for graph

            q_torch2 = qubo_dict_to_torch(nx_graph2, gen_q_dict_mis(nx_graph2), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            # Establish pytorch GNN + optimizer
            real=True
            if real:
                n = len(nodes_l)
                q_torch=q_torch2
                nx_graph=nx_graph2
                graph_dgl=graph_dgl2

            opt_params = {'lr': learning_rate}
            gnn_hypers = {
                'dim_embedding': dim_embedding,
                'hidden_dim': hidden_dim,
                'dropout': 0.0,
                'number_classes': 1,
                'prob_threshold': PROB_THRESHOLD,
                'number_epochs': number_epochs,
                'tolerance': tol,
                'patience': patience
            }

            net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)

            # For tracking hyperparameters in results object
            gnn_hypers.update(opt_params)

            print('Running GNN...')
            gnn_start = time()

            neto, epoch, final_bitstring, best_bitstring = run_gnn_training(
                q_torch, graph_dgl, net, embed, optimizer, gnn_hypers['number_epochs'],
                gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'], file, params)

            gnn_time = time() - gnn_start

            final_loss = loss_func(final_bitstring.float(), q_torch)

            final_bitstring_str = ','.join([str(x) for x in final_bitstring])

            # Process bitstring reported by GNN
            size_mis, ind_set, number_violations = postprocess_gnn_mis(best_bitstring, nx_graph)
            gnn_tot_time = time() - gnn_start

            print(f'Independence number found by GNN is {size_mis} with {number_violations} violations')
            print(f'Took {round(gnn_tot_time, 3)}s, model training took {round(gnn_time, 3)}s')




