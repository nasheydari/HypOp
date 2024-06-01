import numpy as np
import torch
import timeit
from src.loss import loss_sat_numpy, loss_maxcut_numpy, loss_maxcut_numpy, loss_maxind_numpy, loss_maxind_QUBO, loss_task_numpy, loss_task_numpy_vec, loss_mincut_numpy, loss_partition_numpy
import random
import networkx as nx
import copy

from collections import OrderedDict, defaultdict

# Shared Functions
def generate_H_from_constraints(constraints, main, self_loop=False):
    """
    Generate the hypgraph incidence matrix H from hyper constriants list
    :param edges: Hyper edges. List of nodes that in that hyper edges.
    :n: number of nodes
    :self_loop: Whether need to add self_loops. 
    """
    H = []
    i = 1
    dct = {}
    new_constraints = []
    dct[main] = 0
    for c in constraints:
        temp = []
        for node in c:
            if abs(node) not in dct:
                dct[abs(node)] = i
                i += 1
            temp.append(dct[abs(node)])
        new_constraints.append(temp)
    n = len(dct)
    for c in new_constraints:
        temp = [0 for j in range(n)]
        for node in c:
            temp[node] = 1
        H.append(temp)
    if self_loop:
        # added self loop hyper edges
        for i in range(n):
            temp = [0 for j in range(n)]
            temp[i] = 1
            H.append(temp)
    return np.array(H, dtype=float).T, dct

def generate_H_from_edges(edges, n, self_loop=False):
    """
    Generate the hypgraph incidence matrix H from hyper edges list
    :param edges: Hyper edges. List of nodes that in that hyper edges.
    :n: number of nodes
    :self_loop: Whether need to add self_loops. 
    """
    H = []

    for edge in edges:
        temp = [0 for j in range(n)]
        for node in edge:
            temp[node] = 1
        H.append(temp)
    if self_loop:
        # added self loop hyper edges
        for i in range(n):
            temp = [0 for j in range(n)]
            temp[i] = 1
            H.append(temp)
    Ht=np.array(H, dtype=float).T
    return Ht

def _generate_G_from_H(H, variable_weight=False):
    """
    This function is implemented by Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong, Ji, Yue Gao from Xiamen University and Tsinghua University
    Originally github repo could be found here https://github.com/iMoonLab/HGNN
    Originally paper could be found here https://arxiv.org/abs/1809.09401
    
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    #H = np.array(H)
    n_edge = H.shape[1]
    n_node = H.shape[0]
    #Adjacency matrix of the graph with self loop
    A = get_adj(H, n_node, n_edge)
    DA=np.sum(A, axis=1)
    invDA = np.mat(np.diag(np.power(DA, -0.5)))

    Ga=invDA @ A @ invDA

    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)

    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    DEm = DE-1
    inDEm = np.mat(np.diag(np.power(DEm, -1)))
    inDEm = np.nan_to_num(inDEm, 0)
    invDE = np.mat(np.diag(np.power(DE, -1)))


    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T


    #G = H * W * invDE * HT
    Hp = H @ inDEm @ HT
    Hp = Hp - np.diag(np.diagonal(Hp))

    G = DV2 @ H @ W @ invDE @ HT @ DV2
    #G = DV2 @ H @ HT @ DV2
    Gp = DV2 @ Hp @ DV2

    return Gp

def get_normalized_G_from_con(constraints, header):
    n_nodes = header['num_nodes']
    G_matrix =  torch.zeros(n_nodes, n_nodes)
    indegree = [0] * n_nodes
    outdegree = [0] * n_nodes
    for (u, v) in constraints:
        indegree[u-1] += 1
        outdegree[u-1] += 1
        indegree[v-1] += 1
        outdegree[v-1] += 1
    for (u, v) in constraints:
        u_ = u - 1
        v_ = v - 1
        G_matrix[u_][v_] += 1 * (indegree[u_] ** (-0.5)) * (outdegree[u_] ** (-0.5))
        G_matrix[v_][u_] += 1 * (indegree[v_] ** (-0.5)) * (outdegree[v_] ** (-0.5))
    #for u in range(0, n_nodes):
    #    G_matrix[u][u] += 1
    return G_matrix

def get_adj(H, n_nodes, n_edges):
    A=np.zeros([n_nodes,n_nodes])
    for i in range(n_nodes):
        edges=np.argwhere(H[i, :] > 0).T[0]
        for e in edges:
            nodes=np.argwhere(H[:, e] > 0).T[0]
            for j in nodes:
                A[i,j]=1
    return A

def samples_to_Xs(sample, n, f):
    Xs = {}
    for i in range(n):
        a , b = i*f, (i+1)*f
        Xs[i+1] = torch.from_numpy(sample[a:b][None]).float()
    return Xs

def find_pres(out, tar):
    cur = out[tar]
    pres = []
    for node in out.keys():
        if node != tar:
            if out[node] > cur:
                pres.append(node)
    return pres

def all_to_weights(weights_all, n, C):
    weights = {x+1: [] for x in range(n)}
    for c, w in zip(C, weights_all):
        for node in c:
            weights[abs(node)].append(w)
    return weights

def all_to_weights_task(weights_all, n, C):
    weights = {x+1: [] for x in range(n)}
    for c, w in zip(C, weights_all):
        for node in c[:-1]:
            weights[abs(node)].append(w)
    return weights

# mapping functions
def mapping_algo(best_outs, weights, info, mode):
    finished = set()
    pres = {x: find_pres(best_outs, x) for x in best_outs.keys()}
    res = {x: 0 for x in best_outs.keys()}
    n = len(best_outs.keys())
    if mode == 'sat':
        _loss = loss_sat_numpy
    elif mode == 'maxcut':
        _loss = loss_maxcut_numpy
    while len(finished) < n:
        this_round = []
        for i in pres:
            if all([x in finished for x in pres[i]]):
                temp = res.copy()
                temp[i] = 1
                if _loss(temp, info[i], weights[i]) < _loss(res, info[i], weights[i]):
                    res = temp
                finished.add(i)
                this_round.append(i)
        for ele in this_round:
            del pres[ele]
    return res

# def mapping_distribution(best_outs, params, n, info, weights, constraints, all_weights, inc, penalty, hyper):
#     Nei=Neighbors(n, info)
#     best_score = float('inf')
#     lb = float('inf')
#     if params['mode'] == 'sat':
#         _loss = loss_sat_numpy
#     elif params['mode'] == 'maxcut':
#         _loss = loss_maxcut_numpy
#     elif params['mode'] == 'maxind':
#         _loss = loss_maxind_numpy
#     for rea in range(params['N_realize']):
#         res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
#         l1 =_loss(res, constraints, all_weights, hyper=hyper)
#         l1i=np.zeros([n,])
#         for i in range(1,n+1):
#             l1i[i-1]=_loss(res, info[i], weights[i], penalty=penalty, hyper=hyper)
#         ord = random.sample(range(1, n + 1), n)
#         t = 1
#         for it in range(params['Niter_h']):
#             print(it)
#             for i in ord:
#                 temp = res.copy()
#                 # temp = pr.copy()
#                 if res[i] == 0:
#                     temp[i] = 1
#                 else:
#                     temp[i] = 0
#                 lt= _loss(temp, info[i], weights[i], penalty=penalty, hyper=hyper)
#                 #lt = _loss(temp, constraints, all_weights, hyper=hyper)
#                 if lt < l1i[i-1] or np.exp(- (lt - l1i[i-1]) / t) > np.random.uniform(0, 1):
#                     l1 = l1 - l1i[i - 1] + lt
#                     l1i[i-1] = lt
#                     res = temp
#                     for nei in Nei[i]:
#                         l1i[nei-1] = _loss(res, info[nei], weights[nei], penalty=penalty, hyper=hyper)
#                     if l1 < lb:
#                         lb = l1
#                         best = res
#             t = t * 0.85
#         score = lb
#         if score < best_score:
#             best_res = best
#             best_score = score
#     return best_res


def mapping_distribution(best_outs, params, n, info, weights, constraints, all_weights, inc, penalty, hyper):
    if params['random_init']=='one_half':
        best_outs= {x: 0.5 for x in best_outs.keys()}
    elif params['random_init']=='uniform':
        best_outs = {x: np.random.uniform(0,1) for x in best_outs.keys()}
    elif params['random_init'] == 'threshold':
        best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

    best_score = float('inf')
    lb = float('inf')
    if params['mode'] == 'sat':
        _loss = loss_sat_numpy
    elif params['mode'] == 'maxcut' or params['mode'] == 'QUBO_maxcut' or params['mode'] == 'maxcut_annea':
        _loss = loss_maxcut_numpy
    elif params['mode'] == 'maxind' or params['mode'] == 'QUBO':
        _loss = loss_maxind_numpy
    elif params['mode'] == 'task':
        _loss = loss_task_numpy
    elif params['mode'] == 'mincut':
        _loss = loss_mincut_numpy

    for rea in range(params['N_realize']):
        res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
        best_score = _loss(res, constraints, all_weights, hyper=hyper)
        best_res = copy.deepcopy(res)
        # ord = random.sample(range(1, n + 1), n)
        t = params['t']
        # l1=best_score
        prev_score=best_score
        # stepsize = n//200
        for it in range(params['Niter_h']):
            print(it)
            # temp = copy.deepcopy(res)
            ord = random.sample(range(1, n + 1), n)
            # j=0
            for i in ord:
                # j+=1
                temp = copy.deepcopy(res)
                # temp = pr.copy()
                if res[i] == 0:
                    temp[i] = 1
                else:
                    temp[i] = 0
                # if (j) % stepsize == 0:
                # lt = _loss(temp, constraints, all_weights, penalty=penalty,  hyper=hyper)
                # l1 = _loss(res, constraints, all_weights, penalty=penalty, hyper=hyper)
                lt = _loss(temp, info[i], weights[i], penalty=penalty, hyper=hyper)
                l1 = _loss(res, info[i], weights[i], penalty=penalty, hyper=hyper)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    res = copy.deepcopy(temp)
                    # l1=lt
                    # print(l1)
                    # temp=copy.deepcopy(res)
            t = t * 0.95
            if (it+1)%100==0:
                # score = _loss(res, constraints, all_weights, hyper=hyper)
                score=l1
                if score==prev_score:
                    print('early stopping of SA')
                    break
                else:
                    prev_score = score
                    print(score)
            # score = _loss(res, constraints, all_weights, hyper=hyper)
            # print(score)
            # if score < best_score:
            #     best_res = copy.deepcopy(res)
            #     best_score = score
        score = _loss(res, constraints, all_weights, hyper=hyper)
        print(score)
        if score < best_score:
            best_res =copy.deepcopy(res)
            best_score = score
    return best_res


def mapping_distribution_QUBO(best_outs, params, q_torch, n):
    #best_outs= {x: 0.5 for x in best_outs.keys()}
    #best_outs = {x: np.random.uniform(0,1) for x in best_outs.keys()}
    best_score = float('inf')
    lb = float('inf')
    _loss = loss_maxind_QUBO
    for rea in range(params['N_realize']):
        res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
        ord = random.sample(range(1, n + 1), n)
        t = 0
        for it in range(params['Niter_h']):
            print(it)
            for i in ord:
                temp = res.copy()
                # temp = pr.copy()
                if res[i] == 0:
                    temp[i] = 1
                else:
                    temp[i] = 0
                lt = _loss(torch.Tensor(list(temp.values())), q_torch)
                l1 = _loss(torch.Tensor(list(res.values())), q_torch)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    res = temp
            t = t * 0.95
        score = _loss(torch.Tensor(list(res.values())), q_torch)
        if score < best_score:
            best_res =res
            best_score = score
    return best_res



def mapping_distribution_vec_task(best_outs, params, n, info, constraints, C_dic, all_weights, inc, lenc, leninfo, penalty, hyper):
    if params['random_init']=='one_half':
        best_outs= {x: 0.5 for x in best_outs.keys()}
    elif params['random_init']=='uniform':
        best_outs = {x: np.random.uniform(0,1) for x in best_outs.keys()}
    elif params['random_init'] == 'threshold':
        best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}
    L=len(constraints)
    best_score = float('inf')
    lb = float('inf')

    if params['mode'] == 'task_vec':
        _loss = loss_task_numpy_vec

    for rea in range(params['N_realize']):
        res = {x: [np.random.choice(range(2), p=[1 - best_outs[x][i], best_outs[x][i]]) for i in range(L)] for x in best_outs.keys()}
        res_array = np.array(list(res.values()))
        # lbest = _loss(res, lenc, leninfo)
        lbest = _loss(res_array, lenc, leninfo)
        l1=lbest
        resbest = res.copy()
        # ord = random.sample(range(1, n + 1), n)
        t = params['t']
        for it in range(params['Niter_h']):
            print(it)
            ord = random.sample(range(1, n + 1), n)
            for i in ord:
                #temp = copy.deepcopy(res)
                temp = copy.deepcopy(res_array)
                # temp = pr.copy()
                j=random.sample(range(L), 1)[0]
                # if res[i][j] == 0:
                #     temp[i][j] = 1
                # else:
                #     temp[i][j] = 0
                if res_array[i-1,j] == 0:
                    temp[i-1,j] = 1
                else:
                    temp[i-1,j] = 0
                lt = _loss(temp, lenc, leninfo)
                #l1 = _loss(res, lenc, leninfo)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    # res = copy.deepcopy(temp)
                    #res=temp.copy()
                    if res_array[i-1,j] == 0:
                        res_array[i-1,j] = 1
                    else:
                        res_array[i-1,j] = 0
                    l1=lt
                    if l1==0:
                        break
                    # if l1<=lbest:
                    #     lbest=l1
                    #     resbest=res.copy()
            if l1 == 0:
                break
            t = t * 0.95
        # score = _loss(res, lenc, leninfo)
        lbest=l1
        score = lbest
        print(score)
        if score <= best_score:
            #best_res =resbest.copy()
            best_res = copy.deepcopy(res_array)
            best_score = score
    return best_res


def mapping_distribution_vec(best_outs, params, n, info, weights, constraints, all_weights, inc, L, penalty,hyper):
    if params['random_init'] == 'one_half':
        best_outs = {x: 0.5 for x in best_outs.keys()}
    elif params['random_init'] == 'uniform':
        best_outs = {x: np.random.uniform(0, 1) for x in best_outs.keys()}
    elif params['random_init'] == 'threshold':
        best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

    best_score = float('inf')
    lb = float('inf')

    if params['mode'] == 'partition':
        _loss = loss_partition_numpy

    for rea in range(params['N_realize']):
        # res={x:best_outs[x].argmax()}
        res={}
        for x in best_outs.keys():
            part=np.random.choice(range(params['n_partitions']), p=best_outs[x])
            res_x=[0 for _ in range(params['n_partitions'])]
            res_x[part]=1
            res[x]=res_x
        # res = {x: [np.random.choice(range(2), p=[1 - best_outs[x][i], best_outs[x][i]]) for i in range(L)] for x in
        #        best_outs.keys()}
        res_array = np.array(list(res.values()))
        # lbest = _loss(res, lenc, leninfo)
        lbest = _loss(res_array, constraints, weights, hyper)
        l1 = lbest
        resbest = res.copy()
        # ord = random.sample(range(1, n + 1), n)
        t = params['t']
        for it in range(params['Niter_h']):
            print(it)
            ord = random.sample(range(1, n + 1), n)
            for i in ord:
                # temp = copy.deepcopy(res)
                temp = copy.deepcopy(res_array)
                # temp = pr.copy()
                temp[i-1,:]=[0 for _ in range(params['n_partitions'])]
                j = random.sample(range(L), 1)[0]
                # if res[i][j] == 0:
                #     temp[i][j] = 1
                # else:
                #     temp[i][j] = 0
                # if res_array[i - 1, j] == 0:
                temp[i - 1, j] = 1
                # else:
                #     temp[i - 1, j] = 0
                lt = _loss(temp, constraints, weights, hyper)
                # l1 = _loss(res, lenc, leninfo)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    # res = copy.deepcopy(temp)
                    # res=temp.copy()
                    res_array[i-1,:] = [0 for _ in range(params['n_partitions'])]
                    res_array[i - 1, j] = 1
                    # if res_array[i - 1, j] == 0:
                    #     res_array[i - 1, j] = 1
                    # else:
                    #     res_array[i - 1, j] = 0
                    l1 = lt
                    # if l1 == 0:
                    #     break
                    # if l1<=lbest:
                    #     lbest=l1
                    #     resbest=res.copy()

                # if sum(res_array[i-1,:])==0 or sum(res_array[i-1,:])>1:
                #     res_array[i - 1, :]=0
                #     arg1=random.randint(0, L-1)
                #     res_array[i - 1, :] = 1
            # if l1 == 0:
            #     break
            t = t * 0.95
        # score = _loss(res, lenc, leninfo)
        lbest = l1
        score = lbest
        print(score)
        if score <= best_score:
            # best_res =resbest.copy()
            best_res = copy.deepcopy(res_array)
            best_score = score
    return best_res

# def mapping_distribution(best_outs, params, n, info, weights, constraints, all_weights, inc, penalty, hyper):
#     best_score = float('inf')
#     lb = float('inf')
#     if params['mode'] == 'sat':
#         _loss = loss_sat_numpy
#     elif params['mode'] == 'maxcut':
#         _loss = loss_maxcut_numpy
#     elif params['mode'] == 'maxind':
#         _loss = loss_maxind_numpy
#     for rea in range(params['N_realize']):
#         res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
#         l1 =_loss(res, constraints, all_weights, hyper=hyper)
#         ord = random.sample(range(1, n + 1), n)
#         t = 1
#         for it in range(params['Niter_h']):
#             print(it)
#             for i in ord:
#                 temp = res.copy()
#                 # temp = pr.copy()
#                 if res[i] == 0:
#                     temp[i] = 1
#                 else:
#                     temp[i] = 0
#                 lt = _loss(temp, constraints, all_weights, hyper=hyper)
#                 if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
#                     l1 = lt
#                     res = temp
#                     if l1 < lb:
#                         lb = l1
#                         best = res
#             t = t * 0.85
#         score = lb
#         if score < best_score:
#             best_res = best
#             best_score = score
#     return best_res

def Neighbors(n, info):
    Nei= {}
    for i in range(1,n+1):
        ne=[]
        for x in info[i]:
            for j in x:
                ne.append(j)
        ne=set(ne)
        ne.discard(i)
        Nei[i]=ne
    return Nei

import h5py
import pandas as pd
def analysis_res(path):
    with h5py.File(path, 'r') as f:
        names = []
        reses = []
        for key in f.keys():
            names.append(key)
            reses.append(f[key][:][0])
    res  = pd.DataFrame()
    res['File_name'] = names
    res['Result'] = reses        
    return res


def gen_q_mis(constraints, n_nodes, penalty=2 ,torch_dtype=None, torch_device=None):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.

    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_mat = torch.zeros(n_nodes, n_nodes)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for cons in constraints:
        Q_mat[cons[0]-1][cons[1]-1] = penalty
        Q_mat[cons[1] - 1][cons[0] - 1] = penalty
    # all diagonal terms get -1
    for u in range(n_nodes):
        Q_mat[u][u] = -1


    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)


    return Q_mat

def gen_q_maxcut(constraints, n_nodes,torch_dtype=None, torch_device=None):
    """
    Helper function to generate QUBO matrix for Maxcut as minimization problem.

    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_mat = torch.zeros(n_nodes, n_nodes)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for cons in constraints:
        Q_mat[cons[0]-1][cons[1]-1] = 1
        Q_mat[cons[1] - 1][cons[0] - 1] = 1
    # all diagonal terms get -1
    for u in range(n_nodes):
        Q_mat[u][u] = -1


    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)


    return Q_mat


def Maxind_postprocessing(res, constraints,n):
    res_copy=res
    graph_p = nx.Graph()
    graph_p.add_nodes_from(range(1,n+1))
    graph_p.add_edges_from(constraints)
    n=len(res)
    nei={}
    score={}
    for i in range(1,n+1):
        if res[i]==1:
            nei[i]=list(graph_p.neighbors(i))
            score[i]=sum([res[item] for item in nei[i]])
        else:
            score[i] = 0
    score_s=sorted(score.items(), key=lambda x: x[1], reverse=True)
    score_sd = {id: jd for (id, jd) in score_s}
    ss=0
    for cons in constraints:
        ss += res[cons[0]] * res[cons[1]]
    print(ss)

    while sum(score_sd.values())>0:
        nodes=list(score_sd.keys())
        res[nodes[0]]=0
        score[nodes[0]] = score[nodes[0]]-1
        score_s = sorted(score.items(), key=lambda x: x[1], reverse=True)
        score_sd = {id: jd for (id, jd) in score_s}

    return res_copy


def sparsify_graph(constraints, header,  info, spars_p):
    n=header['num_nodes']
    m=header['num_constraints']
    constraints2=copy.deepcopy(constraints)
    info2=copy.deepcopy(info)
    for edge in constraints:
        n1=edge[0]
        n2=edge[1]
        if len(info2[n1])>1 and len(info2[n2])>1:
            rnd=np.random.uniform(0, 1)
            if rnd<spars_p:
                constraints2.remove(edge)
                info2[n1].remove(edge)
                info2[n2].remove(edge)
    header2={}
    header2['num_nodes']=n
    header2['num_constraints']=len(constraints2)
    return constraints2, header2, info2


def generate_watermark(N, wat_len, wat_seed_value):
    # random.seed(wat_seed_value)
    p=0.2
    selected_nodes=random.sample(range(1,N),wat_len)
    Gr = nx.erdos_renyi_graph(len(selected_nodes), p, seed=wat_seed_value, directed=False)

    mapping = {i: node for i, node in enumerate(selected_nodes)}
    Gr = nx.relabel_nodes(Gr, mapping)
    wat_G = np.zeros([len(Gr.edges)+1, 2]).astype(np.int64)
    wat_G[0,0]=[wat_len, len(Gr.edges)]
    wat_G[1:,:]=[list(edge) for edge in Gr.edges]

    return wat_G, selected_nodes

