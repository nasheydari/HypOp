from src.utils import generate_H_from_edges, _generate_G_from_H, generate_H_from_constraints, all_to_weights, all_to_weights_task, gen_q_mis, get_normalized_G_from_con, Maxind_postprocessing, sparsify_graph
import numpy as np
import torch
from src.timer import Timer
import timeit
from src.trainer import  centralized_train, GD_train, centralized_train_for, centralized_train_vec, centralized_train_att, centralized_train_bipartite, centralized_train_cliquegraph, centralized_train_coarsen
from src.loss import loss_maxcut_numpy_boost, loss_sat_numpy_boost, loss_maxind_numpy_boost, loss_maxind_QUBO, loss_task_numpy, loss_task_numpy_vec, loss_mincut_numpy_boost, loss_watermark, loss_partition_numpy, loss_partition_numpy_boost
import matplotlib.pyplot as plt
import dgl
import random
import os
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
# from dgl.nn.pytorch import GraphConv
from itertools import chain, islice, combinations
from networkx.algorithms import maximal_independent_set as mis
from time import time
from src.data_reading import read_uf, read_stanford, read_hypergraph
import logging
from src.QUBO_utils import generate_graph, get_gnn, run_gnn_training, qubo_dict_to_torch, gen_combinations, loss_func
from src.coarsen import coarsen1, coarsen2, coarsen3, coarsen4, coarsen5



#### main solver ####
def centralized_solver(constraints, header, params, file_name):

    temp_time = timeit.default_timer()



    if params['coarsen']:
        new_header, new_constraints, graph_dict = coarsen5(constraints, header)
        n_org=header['num_nodes']
        n=new_header['num_nodes']
        ##### q_torch helps with calculating the Maxcut and MIS loss faster ####
        q_torch = gen_q_mis(constraints, n_org, 2, torch_dtype=None, torch_device=None)
    else:
        graph_dict = {}
        new_header, new_constraints = header, constraints
        n = header['num_nodes']
        ##### q_torch helps with calculating the Maxcut and MIS loss faster ####
        q_torch = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)

    # if params['data'] != 'hypergraph' and  params['data'] != 'task' and params['data'] != 'uf' and params['data'] != 'NDC':




    if params['f_input']:
        f=params['f']
    else:
        f = int(np.sqrt(n))


    if params['data'] == 'bipartite':
        path = params['folder_path_hyper'] + file_name[:-14]+'.txt'
        constraints_hyper, header_hyper = read_hypergraph(path)
        n_hyper=header_hyper['num_nodes']
    elif params['data'] == 'cliquegraph':
        path = params['folder_path_hyper'] + file_name[:-10] + '.txt'
        constraints_hyper, header_hyper = read_hypergraph(path)
        n_hyper = header_hyper['num_nodes']

    if params['coarsen']:
        info = {x + 1: [] for x in range(n_org)}
    else:
        info = {x + 1: [] for x in range(n)}
    for constraint in constraints:
        if params['data'] == 'task':
            for node in constraint[:-1]:
                info[abs(node)].append(constraint)
        else:
            for node in constraint:
                info[abs(node)].append(constraint)

    if params['data'] == 'bipartite' or params['data'] == 'cliquegraph':
        info_hyper = {x + 1: [] for x in range(n_hyper)}
        for constraint in constraints_hyper:
            if params['data'] == 'task':
                for node in constraint[:-1]:
                    info_hyper[abs(node)].append(constraint)
            else:
                for node in constraint:
                    info_hyper[abs(node)].append(constraint)

    sparsify = params['sparcify']
    spars_p=params['sparcify_p']
    if sparsify:
        if not params['coarsen']: #### we don't do sparcify with coarsen
            constraints_sparse, header_sparse, info_sparse = sparsify_graph(constraints, header, info, spars_p)
        else:
            constraints_sparse, header_sparse, info_sparse = new_constraints, new_header, info

    elif params['coarsen']:
        constraints_sparse, header_sparse, info_sparse = new_constraints, new_header, info
    else:
        constraints_sparse, header_sparse, info_sparse = constraints, header, info


    if params['data']!='task':
        edges = [[abs(x) - 1 for x in edge] for edge in constraints_sparse]
    else:
        edges = [[abs(x) - 1 for x in edge[:-1]] for edge in constraints_sparse]

    ###### construct the G matrix, which is the HyperGNN or GNN aggregation matrix ####


    #load = False   ###### if we have saved G before and want to load it ######
    load= False
    #### if params['random_init']==true, it means we are not going to use the HyperGNN training results and use SA with random initialization ######
    if params['random_init']=='none' and not load:
        ##### for GNN (graph problems) #######
        if params['data'] != 'hypergraph' and params['data'] != 'task' and params['data'] != 'uf' and params['data'] != 'NDC':
            G = get_normalized_G_from_con(constraints_sparse, header_sparse)

        ###### for HyperGNN (hypergraph problems) ######
        else:
            H = generate_H_from_edges(edges, n)
            G = _generate_G_from_H(H)
            name_g="./models/G/"+params['mode']+'_'+file_name[:-4]+".npy"
            with open(name_g, 'wb') as ffff:
                np.save(ffff,G)
            G = torch.from_numpy(G).float()

    elif load:
        name_g="./models/G/"+params['mode']+'_'+file_name[:-4]+".npy" #### will have to add to the config file #####
        G=np.load(name_g)
        G = torch.from_numpy(G).float()

    else:
        G=torch.zeros([n,n])

    #### here, we assign weights to constraints in case they have different importance, default=all 1 #####
    # all_weights = [1.0 for c in (constraints)]
    #
    # if params['data'] != 'task':
    #     weights = all_to_weights(all_weights, n, constraints)
    # else:
    #     weights = all_to_weights_task(all_weights, n, constraints)


    #initialization
    reses = []
    reses_th = []
    probs = []
    train_times = []
    map_times = []
    if params['mode'] == 'task_vec':
        L = len(constraints)
    else:
        L = params['n_partitions']

    for i in range(params['K']): ## if we want to solve the problem for K number of times ##

        if params["mode"]=='task_vec': ## resource allocation problem ##
            if params['mode'] == 'task_vec':
                C_dic = {}
                ic = 0
                lenc = torch.zeros([L])
                for c in constraints:
                    lenc[ic] = len(c)
                    C_dic[str(c)] = ic
                    ic += 1
                leninfo = torch.zeros([n])
                for inn in range(n):
                    leninfo[inn] = len(info[inn + 1])

            res, prob, train_time, map_time = centralized_train_vec(G, params, constraints, n, info,
                                                                    file_name, L)
            train_times.append(train_time)
            map_times.append(map_time)
        elif params['coarsen']:
            res, prob, train_time, map_time = centralized_train_coarsen(G, params, f, new_constraints, constraints, graph_dict,n_org, n, info,
                                                                    file_name)

        elif params['mode']=='partition' or params['mode']=='MNP':
            if params['mode']=='MNP':
                L = params['n_knapsacks']+1
            res, prob, train_time, map_time = centralized_train_vec(G, params, constraints, n, info,
                                                                    file_name, L)

            train_times.append(train_time)
            map_times.append(map_time)
        elif params["data"] == 'bipartite': ##### bipartite GNN #####
            res, prob, train_time, map_time = centralized_train_bipartite(G, params, f, constraints_hyper, n,
                                                                          n_hyper, info_hyper, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        elif params["data"] == 'cliquegraph': ##### bipartite GNN #####
            res, prob, train_time, map_time = centralized_train_cliquegraph(G, params, f, constraints_hyper, n, info_hyper, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        elif not params["GD"] and not params["Att"]: ####### Main Trainer #######
            res,  prob , train_time, map_time= centralized_train( G, params, f, constraints, n, info, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        elif params["Att"]: ###### Trainer with Hypergraph Attention Network ######
            res, prob, train_time, map_time = centralized_train_att(H, params, f, constraints, n, info, file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        else: ###### Gradient Descent Solver (no HyperGNN) #####
            res, prob, train_time, map_time = GD_train (params, f, constraints, n, info,  file_name)
            train_times.append(train_time)
            map_times.append(map_time)

        ##### get the HyperGNN solution with a threshold mapping (res_th) (no fine-tuning) #####
        if params["mode"]!='task_vec' and params["mode"]!='partition':
            res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}
        elif params["mode"]=='partition':
            res_th={}
            for x in range(n):
                max_index = np.argmax(prob[x,:])
                # Create a new array of zeros with the same shape as the original vector
                result = [0 for l in range(L)]
                # Assign 1 to the element at the maximum index
                result[max_index] = 1
                res_th[x]=result
            # res_th = {x: [0 if prob[x,i]<0.5 else 1 for i in range(L)] for x in range(n)}
        else:
            res_th = {x: [0 if prob[x, i] < 0.5 else 1 for i in range(L)] for x in range(n)}

        ##### calculate the score of the fine-tuned result (res) and threshold mapping result (res_th) #######
        if params['mode'] == 'sat':
            score, new_w = loss_sat_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
            score_th, new_w = loss_sat_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],                                 inc=params['boosting_mapping'])

        elif params['mode'] == 'maxcut' or params['mode'] == 'QUBO_maxcut' or params['mode'] == 'maxcut_annea':
            if params['data']=='bipartite' or params['data']=='cliquegraph':
                score, new_w = loss_maxcut_numpy_boost(res, constraints_hyper, [1 for i in range(len(constraints_hyper))],
                                                       inc=params['boosting_mapping'])
                score_th, _ = loss_maxcut_numpy_boost(res_th, constraints_hyper, [1 for i in range(len(constraints_hyper))],
                                                      inc=params['boosting_mapping'])
            else:
                score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                score_th, _ =  loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])

        elif params['mode'] == 'maxind':
            if params["coarsen"]:
                res_feas = Maxind_postprocessing(res, constraints, n_org)
                res_th_feas = Maxind_postprocessing(res_th, constraints, n_org)
            else:
                res_feas = Maxind_postprocessing(res, constraints, n)
                res_th_feas = Maxind_postprocessing(res_th, constraints, n)


            score, score1, new_w = loss_maxind_numpy_boost(res_feas, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
            score_th, score1, new_w = loss_maxind_numpy_boost(res_th_feas, constraints, [1 for i in range(len(constraints))],inc=params['boosting_mapping'])
            #score2, score12, new_w2 = loss_maxind_numpy_boost(res2, constraints, [1 for i in range(len(constraints))],inc=params['boosting_mapping'])
            #scores2.append(score2)

        elif params['mode'] == 'QUBO': #### faster maxind (MIS) problem ####
            if params["coarsen"]:
                res_feas = Maxind_postprocessing(res, constraints, n_org)
                res_th_feas = Maxind_postprocessing(res_th, constraints, n_org)
            else:
                res_feas = Maxind_postprocessing(res, constraints, n)
                res_th_feas = Maxind_postprocessing(res_th, constraints, n)

            score = loss_maxind_QUBO(torch.Tensor(list(res_feas.values())), q_torch)
            # score_old, score1, _ = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
            #                                                inc=params['boosting_mapping'])
            score_th = loss_maxind_QUBO(torch.Tensor(list(res_th_feas.values())), q_torch)

            #score2 = loss_maxind_QUBO(torch.Tensor(list(res2.values())), q_torch)
            #scores2.append(score2)

        elif params['mode'] == 'task': ### I think this doesn't work and for resource allocation, solve task_vec #####
            #res_feas = Task_postprocessing(res, constraints, n)
            #res_th_feas = Maxind_postprocessing(res_th, constraints, n)
            score = loss_task_numpy(res,constraints, [1 for i in range(len(constraints))] , penalty=0, hyper=False)
            # score_old, score1, _ = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
            #                                                inc=params['boosting_mapping'])
            score_th = loss_task_numpy(res_th,constraints , [1 for i in range(len(constraints))], penalty=0, hyper=False)

        elif params['mode'] == 'task_vec':
            # res_m=np.zeros([n,len(constraints)])
            # for i in range(n):
            #     res_m[i,:]=res[i+1]
            leninfon = torch.Tensor.numpy(leninfo)
            lencn = torch.Tensor.numpy(lenc)
            score = loss_task_numpy_vec(res, lencn, leninfon)
            res_th_array = np.array(list(res_th.values()))
            score_th = loss_task_numpy_vec(res_th_array, lencn, leninfon)

        elif params['mode'] == 'mincut':
            score_im, score_cut, new_w = loss_mincut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
            score_th_im,score_th_cut, _ =  loss_mincut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
            score=[score_im, score_cut]
            score_th=[score_th_im,score_th_cut]
        elif params['mode'] == 'partition':
            res_th=np.array(list(res_th.values()))
            score_im, score_cut = loss_partition_numpy_boost(res, constraints, [1 for i in range(len(constraints))], L, params['hyper'])
            score_th_im, score_th_cut= loss_partition_numpy_boost(res_th, constraints,
                                                                   [1 for i in range(len(constraints))],L,
                                                                   params['hyper'])

            score = [score_im, score_cut]
            score_th = [score_th_im, score_th_cut]

        ##### collect the results for each k #####
        probs.append(prob)

        reses.append(score)
        reses_th.append(score_th)

    return reses, reses_th, res,res_th, probs, timeit.default_timer() - temp_time, train_times, map_times

#### solver for multi-gpu (distributed) training ####
def centralized_solver_for(constraints, header, params, file_name, device=0,
                           cur_nodes=None, inner_constraint=None, outer_constraint=None):
    temp_time = timeit.default_timer()
    edges = [[abs(x) - 1 for x in edge] for edge in constraints]
    if cur_nodes is None:
        n = header['num_nodes']
    else:
        n = len(cur_nodes)

    f = int(np.sqrt(n))
    # f=n // 2

    info = {x + 1: [] for x in range(header['num_nodes'])}
    inner_info = {x: [] for x in cur_nodes}
    outer_info = {x: [] for x in cur_nodes}

    if cur_nodes is None:
        for constraint in constraints:
            for node in constraint:
                info[abs(node)].append(constraint)
    else:
        for constraint in inner_constraint:
            for node in constraint:
                inner_info[abs(node)].append(constraint)
        for constraint in outer_constraint:
            for node in constraint:
                if node in cur_nodes:
                    outer_info[abs(node)].append(constraint)

        for constraint in constraints:
            for node in constraint:
                info[abs(node)].append(constraint)
    all_weights = [[1.0 for c in (inner_constraint)] for i in range(params['num_samples'])]

    weights = [all_to_weights(all_weights[i], header['num_nodes'], inner_constraint) for i in range(len(all_weights))]
    # sampler initialization
    sampler = Sampler(params, n, f)
    # timer initialization
    reses = []
    reses2 = []
    reses_th = []
    probs = []
    for i in range(params['K']):
        scores = []
        scores2 = []
        scores_th = []
        scores1 = []
        Xs = sampler.get_Xs(i)
        temp_weights = []
        for j in range(params['num_samples']):
            res, prob, train_time, map_time = centralized_train_for(Xs[j], params, f, constraints, n, info,
                                                                        weights[i], file_name, device
                                                                        , inner_constraint, outer_constraint, cur_nodes,
                                                                        inner_info, outer_info)

            if torch.distributed.get_rank() == 0:
                res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}
                if params['mode'] == 'sat':
                    score, new_w = loss_sat_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                        inc=params['boosting_mapping'])
                    scores.append(score)
                elif params['mode'] == 'maxcut':
                    score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                           inc=params['boosting_mapping'])
                    score_th, _ = loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],
                                                          inc=params['boosting_mapping'])
                    scores.append(score)
                    scores_th.append(score_th)
                elif params['mode'] == 'maxind':
                    res_feas = Maxind_postprocessing(res, constraints, n)
                    score, score1, new_w = loss_maxind_numpy_boost(res, constraints,
                                                                   [1 for i in range(len(constraints))],
                                                                   inc=params['boosting_mapping'])
                    score_th, score1, new_w = loss_maxind_numpy_boost(res_th, constraints,
                                                                      [1 for i in range(len(constraints))],
                                                                      inc=params['boosting_mapping'])
                    print(score, score1)
                    scores.append(score)
                    scores1.append(score1)
                    scores_th.append(score_th)


            if torch.distributed.get_rank() == 0:
                probs.append(prob)
        if torch.distributed.get_rank() == 0:
            sampler.update(scores)
            reses.append(scores)
            reses_th.append(scores_th)
    if torch.distributed.get_rank() == 0:
        return reses, reses2, reses_th, probs, timeit.default_timer() - temp_time, train_time, map_time
    else:
        return None, None, None, None, None, None, None



def QUBO_solver(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
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
    number_epochs = int(params["epoch"])
    learning_rate = params["lr"]
    PROB_THRESHOLD = 0.5

    # Early stopping to allow NN to train to near-completion
    tol = params["tol"]  # loss must change by more than tol, or trigger
    patience = params["patience"]  # number early stopping triggers before breaking loop

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
            log.info(f'{file}:, running time: {gnn_tot_time}, res_th: {size_mis}')

            print(f'Independence number found by GNN is {size_mis} with {number_violations} violations')
            print(f'Took {round(gnn_tot_time, 3)}s, model training took {round(gnn_time, 3)}s')






def centralized_solver_watermark(constraints, watermark_cons, watermark_nodes, wat_type, header, params, file_name):
    temp_time = timeit.default_timer()

    n = header['num_nodes']
    # if params['data'] != 'hypergraph' and  params['data'] != 'task' and params['data'] != 'uf' and params['data'] != 'NDC':
    q_torch = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)
    if params['f_input']:
        f = params['f']
    else:
        f = int(np.sqrt(n))

    if params['data'] == 'bipartite':
        path = params['folder_path_hyper'] + file_name[:-14] + '.txt'
        constraints_hyper, header_hyper = read_hypergraph(path)
        n_hyper = header_hyper['num_nodes']

    constraints=[con.append("Main") for con in constraints]
    watermark_cons=[con.append("Watermark") for con in watermark_cons]

    Constraints_tot= constraints+watermark_cons
    header_tot=header
    header_tot['num_constraints']=len(Constraints_tot)
    info = {x + 1: [] for x in range(n)}
    for constraint in Constraints_tot:
        if params['data'] == 'task':
            for node in constraint[:-2]:
                info[abs(node)].append(constraint)
        else:
            for node in constraint[:-1]:
                info[abs(node)].append(constraint)

    if params['data'] == 'bipartite':
        info_hyper = {x + 1: [] for x in range(n_hyper)}
        for constraint in constraints_hyper:
            if params['data'] == 'task':
                for node in constraint[:-1]:
                    info_hyper[abs(node)].append(constraint)
            else:
                for node in constraint:
                    info_hyper[abs(node)].append(constraint)


    sparsify = False
    spars_p = 0.8
    if sparsify:
        constraints_sparse, header_sparse, info_sparse = sparsify_graph(constraints, header, info, spars_p)
    else:
        constraints_sparse, header_sparse, info_sparse = Constraints_tot, header_tot, info

    if params['data'] != 'task':
        edges = [[abs(x) - 1 for x in edge[:-1]] for edge in constraints_sparse]
    else:
        edges = [[abs(x) - 1 for x in edge[:-2]] for edge in constraints_sparse]

    # load = False
    load = False
    if params['random_init'] == 'none' and not load:
        if params['data'] != 'hypergraph' and params['data'] != 'task' and params['data'] != 'uf' and params[
            'data'] != 'NDC':
            G = get_normalized_G_from_con(constraints_sparse, header_sparse)
        else:
            H = generate_H_from_edges(edges, n)
            G = _generate_G_from_H(H)
            name_g = "./models/G/" + params['mode'] + '_' + file_name[:-4] + ".npy"
            with open(name_g, 'wb') as ffff:
                np.save(ffff, G)
            G = torch.from_numpy(G).float()
    elif load:
        name_g = "./models/G/" + params['mode'] + '_' + file_name[:-4] + ".npy"
        G = np.load(name_g)
        G = torch.from_numpy(G).float()
    else:
        G = torch.zeros([n, n])

    all_weights = [[1.0 for c in (Constraints_tot)] for i in range(params['num_samples'])]
    if params['data'] != 'task':
        weights = [all_to_weights(all_weights[i], n, Constraints_tot) for i in range(len(all_weights))]
    else:
        weights = [all_to_weights_task(all_weights[i], n, Constraints_tot) for i in range(len(all_weights))]
    # sampler initialization
    sampler = Sampler(params, n, f)
    # timer initialization
    reses = []
    reses2 = []
    reses_th = []
    probs = []
    train_times = []
    map_times = []
    for i in range(params['K']):
        # print(weights)
        scores = []
        scores2 = []
        scores_th = []
        scores1 = []
        Xs = sampler.get_Xs(i)
        temp_weights = []
        for j in range(params['num_samples']):
            # res, res2, prob = centralized_train(Xs[j], Gn, params, f, constraints, n, info, weights[i])
            if params["mode"] == 'task_vec':
                C_dic = {}
                ic = 0
                lenc = torch.zeros([len(Constraints_tot)])
                for c in Constraints_tot:
                    lenc[ic] = len(c)
                    C_dic[str(c)] = ic
                    ic += 1

                leninfo = torch.zeros([n])
                for inn in range(n):
                    leninfo[inn] = len(info[inn + 1])

                res, prob, train_time, map_time = centralized_train_vec(Xs[j], G, params, f, Constraints_tot, n, info,
                                                                        weights[0], file_name, C_dic, lenc)
                train_times.append(train_time)
                map_times.append(map_time)
            elif params["data"] == 'bipartite':
                res, prob, train_time, map_time = centralized_train_bipartite(Xs[j], G, params, f, constraints_hyper, n,
                                                                              n_hyper, info_hyper, weights[0],
                                                                              file_name)
                train_times.append(train_time)
                map_times.append(map_time)
            elif not params["GD"] and not params["Att"]:
                res, prob, train_time, map_time = centralized_train_watermark(Xs[j], G, params, f, constraints, watermark_cons, watermark_nodes, n, info,
                                                                    weights[0], file_name)
                train_times.append(train_time)
                map_times.append(map_time)
            elif params["Att"]:
                res, prob, train_time, map_time = centralized_train_att(Xs[j], H, params, f, constraints, n, info,
                                                                        weights[0], file_name)
                train_times.append(train_time)
                map_times.append(map_time)
            else:
                res, prob, train_time, map_time = GD_train(Xs[j], G, params, f, constraints, n, info, weights[0],
                                                           file_name)
                train_times.append(train_time)
                map_times.append(map_time)
            if params["mode"] != 'task_vec':
                res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}
            else:
                res_th = {x: [0 if prob[x, i] < 0.5 else 1 for i in range(len(Constraints_tot))] for x in range(n)}

            if params['mode'] == 'sat':
                res = wat_postprocessing(res, watermark_nodes, watermark_cons, n)
                score, new_w = loss_sat_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                    inc=params['boosting_mapping'])
                scores.append(score)
                score, new_w = loss_sat_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],
                                                    inc=params['boosting_mapping'])
                scores_th.append(score)
                wat_score=loss_watermark(res,watermark_nodes, watermark_cons, wat_type)
            elif params['mode'] == 'maxcut' or params['mode'] == 'QUBO_maxcut' or params['mode'] == 'maxcut_annea':
                res=wat_postprocessing(res, watermark_nodes, watermark_cons, n)
                res_wat=[res[node] for node in watermark_nodes]
                if params['data'] == 'bipartite':
                    score, new_w = loss_maxcut_numpy_boost(res, constraints_hyper,
                                                           [1 for i in range(len(constraints_hyper))],
                                                           inc=params['boosting_mapping'])
                    score_th, _ = loss_maxcut_numpy_boost(res_th, constraints_hyper,
                                                          [1 for i in range(len(constraints_hyper))],
                                                          inc=params['boosting_mapping'])
                else:
                    score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                           inc=params['boosting_mapping'])
                    score_th, _ = loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],
                                                          inc=params['boosting_mapping'])
                    wat_score = loss_watermark(res_wat, watermark_cons, wat_type)
                scores.append(score)
                scores_th.append(score_th)
            elif params['mode'] == 'maxind':
                res = wat_postprocessing(res, watermark_nodes, watermark_cons, n)
                res= Maxind_postprocessing(res, constraints, n)
                score, score1, new_w = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                               inc=params['boosting_mapping'])
                score_th, score1, new_w = loss_maxind_numpy_boost(res_th, constraints,
                                                                  [1 for i in range(len(constraints))],
                                                                  inc=params['boosting_mapping'])
                wat_score = loss_watermark(res_wat, watermark_cons, wat_type)

                print(score, score1)
                scores.append(score)
                scores1.append(score1)
                scores_th.append(score_th)
                # score2, score12, new_w2 = loss_maxind_numpy_boost(res2, constraints, [1 for i in range(len(constraints))],inc=params['boosting_mapping'])
                # scores2.append(score2)
            elif params['mode'] == 'QUBO':
                res_fes=wat_postprocessing(res, watermark_nodes, watermark_cons, n)
                res_feas = Maxind_postprocessing(res, constraints, n)
                res_th_feas = Maxind_postprocessing(res_th, Constraints_tot, n)
                score = loss_maxind_QUBO(torch.Tensor(list(res_feas.values())), q_torch)
                # score_old, score1, _ = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                #                                                inc=params['boosting_mapping'])
                score_th = loss_maxind_QUBO(torch.Tensor(list(res_th_feas.values())), q_torch)
                scores.append(score)
                scores_th.append(score_th)
                # scores1.append(score1)
                # print(score1)
                # print(score_old)
                # score2 = loss_maxind_QUBO(torch.Tensor(list(res2.values())), q_torch)
                # scores2.append(score2)
                wat_score = loss_watermark(res_wat, watermark_cons, wat_type)
            elif params['mode'] == 'task':
                # res_feas = Task_postprocessing(res, constraints, n)
                # res_th_feas = Maxind_postprocessing(res_th, constraints, n)
                score = loss_task_numpy(res, constraints, [1 for i in range(len(constraints))], penalty=0, hyper=False)
                # score_old, score1, _ = loss_maxind_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                #                                                inc=params['boosting_mapping'])
                score_th = loss_task_numpy(res_th, constraints, [1 for i in range(len(constraints))], penalty=0,
                                           hyper=False)
                scores.append(score)
                scores_th.append(score_th)
            elif params['mode'] == 'task_vec':
                # res_m=np.zeros([n,len(constraints)])
                # for i in range(n):
                #     res_m[i,:]=res[i+1]
                leninfon = torch.Tensor.numpy(leninfo)
                lencn = torch.Tensor.numpy(lenc)
                score = loss_task_numpy_vec(res, lencn, leninfon)
                res_th_array = np.array(list(res_th.values()))
                score_th = loss_task_numpy_vec(res_th_array, lencn, leninfon)
                scores.append(score)
                scores_th.append(score_th)
            elif params['mode'] == 'mincut':
                score_im, score_cut, new_w = loss_mincut_numpy_boost(res, constraints,
                                                                     [1 for i in range(len(constraints))],
                                                                     inc=params['boosting_mapping'])
                score_th_im, score_th_cut, _ = loss_mincut_numpy_boost(res_th, constraints,
                                                                       [1 for i in range(len(constraints))],
                                                                       inc=params['boosting_mapping'])
                scores.append([score_im, score_cut])
                scores_th.append([score_th_im, score_th_cut])
            # score, _ = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
            # scores.append(score)
            probs.append(prob)
        sampler.update(scores)
        reses.append(scores)
        # reses2.append(scores2)
        reses_th.append(scores_th)
    return reses, reses2, reses_th, probs, timeit.default_timer() - temp_time, train_times, map_times

