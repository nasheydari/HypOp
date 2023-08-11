import networkx as nx
import numpy as np
import os
import random
















#path= "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/random_regular/reg_graph_100"
path="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/stanford_data_p1/"
#os.chdir(path)
# for file_name in os.listdir(path):
#     if file_name.startswith('G'):
#         with open(path+file_name) as f:
#             file = f.read()
#         lines = file.split('\n')
#     [n,m]=lines[0][:-2].split(' ')

    # n=int(n)
    # m=int(m)
#n=2000
n=3000
complete=nx.complete_graph(n, create_using=None)
for p in np.linspace(0.001,0.01,10):
    graph_p=nx.Graph()
    graph_p.add_nodes_from(complete)
    for edge in complete.edges:
        rn = random.random()
        if rn<=p:
            graph_p.add_edge(*edge)

    L=len(graph_p.edges)

    G = np.zeros([L+1,2]).astype(np.int64)

    G [0,:] = [int(n), L]

    i=0

    G[1:,:]= graph_p.edges

    G[1:,:] = G[1:,:] + np.ones([L,2])


    #name = 'Random_'+str(int(10*p))+'_'+str(n)+'.txt'
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/Random/3000/"

    np.savetxt(path2+name, G, fmt='%d')


