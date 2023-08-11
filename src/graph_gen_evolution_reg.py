import networkx as nx
import numpy as np
import os
import random










#path= "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/random_regular/reg_graph_100"
path="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/stanford_data_p1_reg/"
#os.chdir(path)
for file_name in os.listdir(path):
    if file_name.startswith('G26'):
        with open(path+file_name) as f:
            file = f.read()
        lines = file.split('\n')
        [n,m]=lines[0].split(' ')
        n = int(n)
        m = int(m)
        constraints = []
        for con in lines[1:-1]:
            temp = con.split(' ')
            constraints.append([int(x) for x in temp[:2]])


        d=int(2*m/n)
        p=2*m/(n*(n-1))
        graph_p = nx.Graph()
        graph_p.add_edges_from(constraints)
        for p in np.linspace(0, 1, 11):
            graph_p_n = nx.Graph.copy(graph_p)
            for edge in graph_p.edges:
                node1=edge[0]
                node2=edge[1]
                node = np.random.choice([node1, node2], 1)[0]
                neighbors=[n for n in graph_p_n.neighbors(node)]
                othernodes=[n for n in graph_p_n.nodes if n not in neighbors]
                node3=np.random.choice(othernodes, 1)[0]
                rn = random.random()
                if rn<=p:
                    graph_p_n.remove_edge(*edge)
                    graph_p_n.add_edge(node,node3)




            L=len(graph_p_n.edges)

            G = np.zeros([L+1,2]).astype(np.int64)

            G [0,:] = [int(n), L]

            i=0

            G[1:,:]= graph_p_n.edges

            G[1:,:] = G[1:,:]


            name = file_name[:-4]+'_Perturb_'+str(int(10*p))+'_'+str(n)+'_'+str(d)+'.txt'
            path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/Reg_perturb/"

            np.savetxt(path2+name, G, fmt='%d')


