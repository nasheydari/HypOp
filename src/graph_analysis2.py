import os

import igraph as ig
import numpy as np
from igraph import Graph
import json
import os
import networkx as nx


G_info={}
path1='/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/graphs4/'
path2="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/G_power_f_dict4.json"
f = open(path2)
dict=json.load(f)



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



        G=nx.Graph()
        G.add_edges_from(constraints)
        Nodes_h=[i for i in G.nodes() if G.degree[i]>5]
        Edges_h = len((G.subgraph(Nodes_h)).edges())
        p_ef=(Edges_h)/(len(Nodes_h)*(len(Nodes_h)-1))
        dict[name]['Closeness_Centrality']=nx.closeness_centrality(G)
        dict[name]["Nodes with degree 6+"]=len(Nodes_h)
        dict[name]["P_ef"] = p_ef
        #dict[name]["Algebraic Connectivity"]=nx.algebraic_connectivity(G)
        # dict[name]['Eigen Vector Centrality']=np.average(list(nx.eigenvector_centrality_numpy(G).values()))
        # dict[name]['is connected']=nx.is_connected(G)
        # dict[name]['diameter']=nx.diameter(G)
        # try:
        #     av_shortest_path = nx.average_shortest_path_length(G)
        # except:
        #     l=[]
        #     for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        #         l.append(nx.average_shortest_path_length(C))
        #     av_shortest_path=np.average(l)
        # dict[name]['Average Shortest Path']=av_shortest_path

with open("/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/G_power_f_dict4_n.json", "w") as write_file:
    json.dump(dict, write_file)


