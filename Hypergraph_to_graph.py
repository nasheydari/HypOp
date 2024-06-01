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
from itertools import combinations



for files in os.listdir('./data/hypergraph_data/synthetic/new/all/'):
    pth = './data/hypergraph_data/synthetic/new/all/'+files
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

        Nodes=list(range(1,header['num_nodes']+1))
        graph = nx.Graph()
        graph.add_nodes_from(Nodes)
        for hyperedge in constraints:
            edges = list(combinations(hyperedge, 2))
            graph.add_edges_from(edges)

        L = len(graph.edges)

        G = np.zeros([L + 1, 2]).astype(np.int64)

        G[0, :] = [header['num_nodes'], L]

        i = 0

        G[1:, :] = graph.edges



        # name = 'Random_'+str(int(10*p))+'_'+str(n)+'.txt'
        name = files[:-4]+'_graph.txt'
        path2 ='./data/hypergraph_data/synthetic/new/graph/'

        np.savetxt(path2 + name, G, fmt='%d')



