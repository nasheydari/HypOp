import json
import numpy as np
# Import Module
import os
import copy
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA

from numpy.random import rand
import random
import csv
import pickle
import networkx as nx



for files in os.listdir('./data/hypergraph_data/synthetic/new/'):
    pth = './data/hypergraph_data/synthetic/new/'+files
    if files.endswith('.txt'):
        with open(pth) as f:
            file = f.read()
        lines = file.split('\n')
        header = {}
        info = lines[0].split(' ')
        header['num_nodes'] = int(info[0])
        header['num_constraints'] = int(info[1])
        constraints = []
        i = 0
        for con in lines[1:-1]:
            temp = con.split(' ')
            constraints.append([int(x) for x in temp])
            i += 1

        Nodes=list(range(1,header['num_nodes']+header['num_constraints']+1))
        graph_bipartite = nx.Graph()
        graph_bipartite.add_nodes_from(Nodes)
        for hyperedge in range(1,len(constraints)+1):
            N_edge=header['num_nodes']+hyperedge
            for j in constraints[hyperedge]:
                edge=(j,N_edge)
                graph_bipartite.add_edge(*edge)

        L = len(graph_bipartite.edges)

        G = np.zeros([L + 1, 2]).astype(np.int64)

        G[0, :] = [header['num_nodes']+header['num_constraints'], L]

        i = 0

        G[1:, :] = graph_bipartite.edges

        G[1:, :] = G[1:, :]

        # name = 'Random_'+str(int(10*p))+'_'+str(n)+'.txt'
        name = files[:-4]+'_bipartite.txt'
        path2 ='./data/hypergraph_data/synthetic/new/bipartite/'

        np.savetxt(path2 + name, G, fmt='%d')



