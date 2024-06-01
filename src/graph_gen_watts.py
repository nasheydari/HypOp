import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt















N=[200]
K=[2,3,4,5,6]
#K=[6]
#P=[0.1,0.5,0.9]
P=[0.5]
#Num=3
Num=1
dic={}
iter=0
for n in N:
    for p in P:
        for k in K:
            iter=0
            while iter<Num:
                try:
                    Gr = nx.watts_strogatz_graph(n, k, p)
                    #d=int(2*len(Gr.edges())/n)
                    pr = 2 * len(Gr.edges()) / (n * (n - 1))
                    print(pr)
                    average_clus=nx.average_clustering(Gr)
                    print("average clustering coeficient: ", average_clus)
                    degree_cor=nx.degree_pearson_correlation_coefficient(Gr)
                    print("degree correlation: ", degree_cor)
                    diameter=nx.diameter(Gr)
                    print("diameter: ", diameter)
                    av_shortest_path=nx.average_shortest_path_length(Gr)
                    print("average shortest path: ", av_shortest_path)
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







                    name = 'G_watts_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(iter)+'.txt'
                    path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/watts/graphs7/"
                    np.savetxt(path2 + name, G, fmt='%d')

                    dic[name]={}
                    dic[name]['Nodes']=n
                    dic[name]['Edges']=L
                    dic[name]['rewiring'] = p
                    dic[name]['k'] = k
                    dic[name]['p']=pr
                    dic[name]['Average Clustering Coefficient']=average_clus
                    dic[name]['Pearson Degree Correlation']=degree_cor
                    dic[name]['Diameter']=diameter
                    dic[name]['Average Shortest Path']=av_shortest_path
                    dic[name]['Average Degree']=Average_degree
                    dic[name]['Eigen Vector Centrality'] = np.average(list(nx.eigenvector_centrality_numpy(Gr).values()))
                    dic[name]['is connected'] = nx.is_connected(Gr)
                    iter += 1
                except:
                    print("skipped")


path = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/watts/"

with open(path+"G_watts_dict7.json", "w") as write_file:
    json.dump(dic, write_file)



# plt.plot(range(Num), [dic[name]['p'] for name in dic])
# plt.ylabel('p')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_p5.png")
# plt.show()
#
#
# plt.plot(range(Num), [dic[name]['Average Clustering Coefficient'] for name in dic])
# plt.ylabel('Average Clustering Coefficient')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_av_clus5.png")
# plt.show()
#
#
# plt.plot(range(Num), [dic[name]['Pearson Degree Correlation'] for name in dic])
# plt.ylabel('Pearson Degree Correlation')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_deg_cor5.png")
# plt.show()
#
#
# plt.plot(range(Num), [dic[name]['Average Degree'] for name in dic])
# plt.ylabel('Average Degree')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_av_deg5.png")
# plt.show()
#
#
# plt.plot(range(Num), [dic[name]['Average Shortest Path'] for name in dic])
# plt.ylabel('Average Shortest Path')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_av_shortpath5.png")
# plt.show()
#
# plt.plot(range(Num), [dic[name]['Diameter'] for name in dic])
# plt.ylabel('Diameter')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_diameter5.png")
# plt.show()
#
#
# plt.plot(range(Num), [dic[name]['Eigen Vector Centrality'] for name in dic])
# plt.ylabel('Eigen Vector Centrality')
# plt.xlabel('i')
# plt.savefig(path+"G_watts_eigcent5.png")
# plt.show()
