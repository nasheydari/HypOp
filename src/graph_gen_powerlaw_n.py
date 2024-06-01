import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt















N=[1000]
Pow=[2]
Num=10
dic={}
iter=0

for n in N:
    for pow in Pow:
        while iter<Num:
            s = nx.utils.powerlaw_sequence(n, pow)  # n nodes, power-law exponent 3
            try:
                Gr = nx.expected_degree_graph(s, selfloops=False)


                #d=int(2*len(Gr.edges())/n)
                p = 2 * len(Gr.edges()) / (n * (n - 1))
                print(p)
                if p>0.005:
                    iter += 1
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







                    name = 'G_power_'+str(n)+'_'+str(pow)+'_'+str(iter)+'.txt'
                    path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/graphs5/"


                    dic[name]={}
                    dic[name]['Nodes']=n
                    dic[name]['Edges']=L
                    dic[name]['Power']=pow
                    dic[name]['p']=p
                    dic[name]['Average Clustering Coefficient']=average_clus
                    dic[name]['Pearson Degree Correlation']=degree_cor
                    # dic[name]['Diameter']=diameter
                    # dic[name]['Average Shortest Path']=av_shortest_path
                    dic[name]['Average Degree']=Average_degree

                    np.savetxt(path2+name, G, fmt='%d')
            except:
                print("skipped")


path = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/powerlawf/"

with open(path+"G_power_f_dict5.json", "w") as write_file:
    json.dump(dic, write_file)



plt.plot(range(Num), [dic[name]['p'] for name in dic])
plt.ylabel('p')
plt.xlabel('i')
plt.savefig(path+"G_power_f_p5.png")
plt.show()


plt.plot(range(Num), [dic[name]['Average Clustering Coefficient'] for name in dic])
plt.ylabel('Average Clustering Coefficient')
plt.xlabel('i')
plt.savefig(path+"G_power_f_av_clus5.png")
plt.show()


plt.plot(range(Num), [dic[name]['Pearson Degree Correlation'] for name in dic])
plt.ylabel('Pearson Degree Correlation')
plt.xlabel('i')
plt.savefig(path+"G_power_f_deg_cor5.png")
plt.show()


plt.plot(range(Num), [dic[name]['Average Degree'] for name in dic])
plt.ylabel('Average Degree')
plt.xlabel('i')
plt.savefig(path+"G_power_f_av_deg5.png")
plt.show()
