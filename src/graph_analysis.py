import os

import igraph as ig
import numpy as np
from igraph import Graph
import json
import os
import networkx as nx


G_info={}
path1='/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/stanford_data_p1/'

for name in os.listdir(path1):
    if name.startswith('G'):
        path=path1+name
        with open(path) as f:
            file = f.read()
        lines = file.split('\n')

        lines=np.array(lines[1:])

        constraints=[]
        for i in range(len(lines)):
            info = lines[i].split(' ')
            info = [x for x in info[0:2] if len(x) > 0]
            constraints.append([int(x) for x in info])

        constraints=constraints[:-1]


        g = Graph(edges=constraints)
        G=nx.Graph()
        G.add_edges_from(constraints)
        G_info[name] = {}
        G_info[name]['number of nodes']=g.vcount()
        G_info[name]['number of edges'] = g.ecount()
        G_info[name]['diameter']=Graph.diameter(g)
        G_info[name]['average_path_length']=Graph.average_path_length(g)
        G_info[name]['clustering_coefficient']=g.transitivity_undirected()
        G_info[name]['average_degree']=np.average(g.degree())
        G_info[name]['std_degree'] = np.std(g.degree())
        G_info[name]['girth']=Graph.girth(g)
        G_info[name]['is connected']=Graph.is_connected(g)
        G_info[name]['MIS']=len(nx.maximal_independent_set(G))

with open("../data/G_info_stanford_p1.json", "w") as write_file:
    json.dump(G_info, write_file)


