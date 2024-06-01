import matplotlib.pyplot as plt
import json
import os
import numpy as np


path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/'
os.chdir(path)

with open('maxcut_example_s2_p1_n.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][3:-2])
    res_th=float(temp3[1][3:-2])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))








with open("maxcut_example.json", "w") as write_file:
    json.dump(log_d, write_file)

path='/Users/nasimeh/Documents/distributed_GCN-main-6/src/'
os.chdir(path)

f=open('G_info_stanford_p1.json')
info=json.load(f)


dict={name: {**info[name], **log_d[name]} for name in info}



dict['G1.txt']['Best_res']=11624
dict['G2.txt']['Best_res']=11620
dict['G3.txt']['Best_res']=11622
dict['G4.txt']['Best_res']=11646
dict['G5.txt']['Best_res']=11631
# dict['G6.txt']['Best_res']=2178
# dict['G7.txt']['Best_res']=2006
# dict['G8.txt']['Best_res']=2005
# dict['G9.txt']['Best_res']=2054
# dict['G10.txt']['Best_res']=2000
# dict['G11.txt']['Best_res']=564
# dict['G12.txt']['Best_res']=556
# dict['G13.txt']['Best_res']=582
dict['G14.txt']['Best_res']=3064
dict['G15.txt']['Best_res']=3050
dict['G16.txt']['Best_res']=3052
dict['G17.txt']['Best_res']=3047
dict['G22.txt']['Best_res']=13359
dict['G23.txt']['Best_res']=13344
dict['G24.txt']['Best_res']=13337
dict['G25.txt']['Best_res']=13340
dict['G26.txt']['Best_res']=13328
# dict['G49.txt']['Best_res']=6000
# dict['G50.txt']['Best_res']=5880
dict['G55.txt']['Best_res']=10294
dict['G70.txt']['Best_res']=9541


for name in dict:
    dict[name]['Ratio']=((dict[name]['res'])/(dict[name]['Best_res']))


x_axis=np.array([dict[name]['diameter'] for name in dict])


y_axis=np.array([dict[name]['Ratio'] for name in dict])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Maxcut')
plt.ylabel('Ratio of HypOp result to the best known')
plt.xlabel('Diameter of the graph')
plt.legend(loc='upper right')
plt.show()



x_axis=np.array([dict[name]['average_degree'] for name in dict])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Maxcut')
plt.ylabel('Ratio of HypOp result to the best known')
plt.xlabel('Average degree of the graph')
plt.legend(loc='lower right')
plt.show()

x_axis=np.array([dict[name]['clustering_coefficient'] for name in dict])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Maxcut')
plt.ylabel('Ratio of HypOp result to the best known')
plt.xlabel('Clustering coefficient of the graph')
plt.legend(loc='upper right')
plt.show()

x_axis=np.array([dict[name]['number of nodes'] for name in dict])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Maxcut')
plt.ylabel('Ratio of HypOp result to the best known')
plt.xlabel('Number of the nodes of the graph')
plt.legend(loc='lower right')
plt.show()





print('finished')



