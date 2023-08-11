import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt














n=200
p=1.4

#path= "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/random_regular/reg_graph_100"
#path="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/stanford_data_p1/"
#os.chdir(path)
for pow in np.linspace(1.5,6,30):
    s = nx.utils.powerlaw_sequence(n, pow)  # n nodes, power-law exponent 3
    Gr = nx.expected_degree_graph(s, selfloops=False)
    d=int(2*len(Gr.edges())/n)
    a = nx.random_regular_graph(d, n, seed=None)
    p = 2 * len(Gr.edges()) / (n * (n - 1))
    print(p)
    ar = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
    print(nx.average_clustering(Gr))
    print(nx.average_clustering(a))
    print(nx.average_clustering(ar))
    print(Gr.nodes())
    print(Gr.edges())

    # draw and show graph
    pos = nx.spring_layout(Gr)
    nx.draw_networkx(Gr, pos)
    plt.show()

    posa = nx.spring_layout(ar)
    nx.draw_networkx(ar, posa)
    plt.show()

    #adj_a = {x: list(dict(a.adj)[x]) for x in dict(a.adj).keys()}

    #keys_s = sorted(adj_a.keys())

    #adj_a_s = {x: adj_a[x] for x in keys_s}


    L = len(Gr.edges())
    Lr=len(ar.edges())
    Lg = len(a.edges())
    G = np.zeros([L+1,2]).astype(np.int64)
    Grr=np.zeros([Lr+1,2]).astype(np.int64)
    Gg = np.zeros([Lg + 1, 2]).astype(np.int64)


    G [0,:] = [int(n), L]



    G[1:,:]= Gr.edges

    G[1:,:] = G[1:,:] + np.ones([L,2])



    ##random
    Grr[0, :] = [int(n), Lr]



    Grr[1:, :] = ar.edges

    Grr[1:, :] = Grr[1:, :] + np.ones([Lr, 2])

    ##regular
    Gg[0, :] = [int(n), Lg]

    Gg[1:, :] = a.edges

    Gg[1:, :] = Gg[1:, :] + np.ones([Lg, 2])



    name = 'G_power_'+str(n)+'_'+str(pow)+'.txt'
    path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlaw2/"
    name2 = 'G_power_random' + str(n) + '_' + str(pow) + '_' + str(p) + '.txt'
    path3 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlaw2/"

    name3 = 'G_power_reg' + str(n) + '_' + str(pow) + '_' + str(d) + '.txt'
    path4 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlaw2/"

    np.savetxt(path2+name, G, fmt='%d')
    np.savetxt(path3+name2, Grr, fmt='%d')
    np.savetxt(path4+name3, Gg, fmt='%d')
