import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt















N=[500]
Pow=[1.3]
Num=10
dic={}
iter=0

for n in N:
    for p in np.linspace(0.005,0.045,10):
        Gr = nx.erdos_renyi_graph(n, p, seed=None, directed=False)


        average_clus=nx.average_clustering(Gr)
        print("average clustering coeficient: ", average_clus)
        degree_cor=nx.degree_pearson_correlation_coefficient(Gr)
        print("degree correlation: ", degree_cor)
        # diameter=nx.diameter(Gr)
        # print("diameter: ", diameter)
        # av_shortest_path=nx.average_shortest_path_length(Gr)
        # print("average shortest path: ", av_shortest_path)
        Average_degree=np.average(Gr.degree())
        print("Average Degree: ", Average_degree)
        # draw and show graph
        # pos = nx.spring_layout(Gr)
        # nx.draw_networkx(Gr, pos)
        # plt.show()


        L = len(Gr.edges())

        G = np.zeros([L+1,2]).astype(np.int64)



        G [0,:] = [int(n), L]



        G[1:,:]= Gr.edges

        G[1:,:] = G[1:,:] + np.ones([L,2])







        name = 'G_random_'+str(n)+'_'+str(p)+'.txt'
        path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/random2/"


        dic[name]={}
        dic[name]['Nodes']=n
        dic[name]['Edges']=L
        dic[name]['p']=p
        dic[name]['Average Clustering Coefficient']=average_clus
        dic[name]['Pearson Degree Correlation']=degree_cor
        dic[name]['Average Degree']=Average_degree

        np.savetxt(path2+name, G, fmt='%d')


path = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/"

with open(path+"G_power_f_dict_random2.json", "w") as write_file:
    json.dump(dic, write_file)



plt.plot(range(Num), [dic[name]['p'] for name in dic])
plt.ylabel('p')
plt.xlabel('i')
plt.savefig(path+"G_power_f_p_r2.png")
plt.show()


plt.plot(range(Num), [dic[name]['Average Clustering Coefficient'] for name in dic])
plt.ylabel('Average Clustering Coefficient')
plt.xlabel('i')
plt.savefig(path+"G_power_f_av_clus_r2.png")
plt.show()


plt.plot(range(Num), [dic[name]['Pearson Degree Correlation'] for name in dic])
plt.ylabel('Pearson Degree Correlation')
plt.xlabel('i')
plt.savefig(path+"G_power_f_deg_cor_r2.png")
plt.show()


plt.plot(range(Num), [dic[name]['Average Degree'] for name in dic])
plt.ylabel('Average Degree')
plt.xlabel('i')
plt.savefig(path+"G_power_f_av_deg_r2.png")
plt.show()
