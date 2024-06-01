import matplotlib.pyplot as plt
import json
import os
import numpy as np
import re


Type='Powerlaw'
path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/'


#logfile='maxind_powerlawf2.log'
logfile='maxind_powerlawf4.log'

dicfile='powerlawf/G_power_f_dict4_n.json'
#dicfile='watts/G_watts_dict7.json'

with open(path+logfile) as f:
    Log=f.readlines()

path2="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/"+dicfile

f = open(path2)
dict=json.load(f)

log_powerlawf=dict

for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    name=temp[0].split(':')[2]
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name = temp[0][10:-1]
    time = float(temp1[1][1:-1])
    res = float(temp2[1][10:-4])
    res_th = float(temp3[1][10:-4])

    log_powerlawf[name]['time']=time
    log_powerlawf[name]['res'] = abs(int(res))
    log_powerlawf[name]['res_th'] = abs(int(res_th))






x_axis1=np.array([log_powerlawf[name]['p'] for name in log_powerlawf])
x_axis2=np.array([log_powerlawf[name]['Average Clustering Coefficient'] for name in log_powerlawf])
x_axis3=np.array([log_powerlawf[name]['Pearson Degree Correlation'] for name in log_powerlawf])
# x_axis4=np.array([log_powerlawf[name]['Average Shortest Path'] for name in log_powerlawf])
# x_axis5=np.array([log_powerlawf[name]['Diameter'] for name in log_powerlawf])
x_axis6=np.array([log_powerlawf[name]['P_ef'] for name in log_powerlawf])
x_axis7=np.array([log_powerlawf[name]['Edges']/log_powerlawf[name]['Nodes'] for name in log_powerlawf])


y_axis1=np.array([log_powerlawf[name]['res_th']/log_powerlawf[name]['res'] for name in log_powerlawf])



nd=len(y_axis1)


plotm=np.zeros([8,nd])

plotm[0,:]=x_axis1
plotm[1,:]=y_axis1
plotm[2,:]=x_axis2
plotm[3,:]=x_axis3
# plotm[4,:]=x_axis4
# plotm[5,:]=x_axis5
plotm[6,:]=x_axis6
plotm[7,:]=x_axis7

plotms=plotm[:, plotm[0].argsort()]
plotms2=plotm[:, plotm[2].argsort()]
plotms3=plotm[:, plotm[3].argsort()]
plotms4=plotm[:, plotm[4].argsort()]
plotms5=plotm[:, plotm[5].argsort()]
plotms6=plotm[:, plotm[6].argsort()]
plotms7=plotm[:, plotm[7].argsort()]


pathfig="/Users/nasimeh/Documents/distributed_GCN-main-6/res/plots/"+Type+'/'

plt.plot(plotms[0,:],plotms[1,:], marker='o', label=Type+'_N=1000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('Density')
plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_p4.png")
# plt.show()

plt.plot(plotms6[6,:],plotms6[1,:], marker='o', label=Type+'_N=1000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('P_ef')
plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_avclus4.png")
plt.show()

# plt.plot(plotms3[3,:],plotms3[1,:], marker='o', label=Type+'_N=1000')
# plt.ylabel('GNN outcome over HypOp outcome')
# plt.xlabel('Pearson Degree Correlation')
# plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_degcor4.png")
# plt.show()


# plt.plot(plotms4[4,:],plotms4[1,:], marker='o', label=Type)
# plt.ylabel('Performance of HypOp for Maxind')
# plt.xlabel('Average Shortest Path')
# plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_avshpath3.png")
# plt.show()

#
# plt.plot(plotms5[5,:],plotms5[1,:], marker='o', label=Type)
# plt.ylabel('Performance of HypOp for Maxind')
# plt.xlabel('Diameter')
# plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_diameter3.png")
# plt.show()


#
# plt.plot(plotms6[6,:],plotms6[1,:], marker='o', label=Type+'_N=1000')
# plt.ylabel('GNN outcome over HypOp outcome')
# plt.xlabel('Eigen Vector Centrality')
# plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_eigcent4.png")
# plt.show()

# plt.plot(plotms7[7,:],plotms7[1,:], marker='o', label=Type+'_N=1000')
# plt.ylabel('GNN outcome over HypOp outcome')
# plt.xlabel('Rewiring Probability*density')
# plt.legend(loc='upper right')
# plt.savefig(pathfig+Type+"_rewiring4.png")
# plt.show()



logfile='maxind_powerlawf2.log'

dicfile='powerlawf/G_power_f_dict2.json'
#dicfile='watts/G_watts_dict7.json'

with open(path+logfile) as f:
    Log=f.readlines()

path2="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/"+dicfile

f = open(path2)
dict=json.load(f)

log_powerlawf=dict

for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    name=temp[0].split(':')[2]
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name = temp[0][10:-1]
    time = float(temp1[1][1:-1])
    res = float(temp2[1][10:-4])
    res_th = float(temp3[1][10:-4])

    log_powerlawf[name]['time']=time
    log_powerlawf[name]['res'] = abs(int(res))
    log_powerlawf[name]['res_th'] = abs(int(res_th))






x_axis1=np.array([log_powerlawf[name]['p'] for name in log_powerlawf])
x_axis2=np.array([log_powerlawf[name]['Average Clustering Coefficient'] for name in log_powerlawf])
x_axis3=np.array([log_powerlawf[name]['Pearson Degree Correlation'] for name in log_powerlawf])
# x_axis4=np.array([log_powerlawf[name]['Average Shortest Path'] for name in log_powerlawf])
# x_axis5=np.array([log_powerlawf[name]['Diameter'] for name in log_powerlawf])
# x_axis6=np.array([log_powerlawf[name]['Eigen Vector Centrality']*log_powerlawf[name]['p'] for name in log_powerlawf])
x_axis7=np.array([log_powerlawf[name]['Edges']/log_powerlawf[name]['Nodes'] for name in log_powerlawf])


y_axis1=np.array([log_powerlawf[name]['res_th']/log_powerlawf[name]['res'] for name in log_powerlawf])



nd=len(y_axis1)


plotm=np.zeros([8,nd])

plotm[0,:]=x_axis1
plotm[1,:]=y_axis1
plotm[2,:]=x_axis2
plotm[3,:]=x_axis3
# plotm[4,:]=x_axis4
# plotm[5,:]=x_axis5
# plotm[6,:]=x_axis6
plotm[7,:]=x_axis7

plotms=plotm[:, plotm[0].argsort()]
plotms2=plotm[:, plotm[2].argsort()]
plotms3=plotm[:, plotm[3].argsort()]
plotms4=plotm[:, plotm[4].argsort()]
plotms5=plotm[:, plotm[5].argsort()]
plotms6=plotm[:, plotm[6].argsort()]
plotms7=plotm[:, plotm[7].argsort()]




plt.plot(plotms[0,:],plotms[1,:], marker='o', label=Type+'_N=500')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('Density')
plt.legend(loc='upper right')


logfile='maxind_powerlawf3.log'

dicfile='powerlawf/G_power_f_dict3.json'
#dicfile='watts/G_watts_dict7.json'

with open(path+logfile) as f:
    Log=f.readlines()

path2="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/"+dicfile

f = open(path2)
dict=json.load(f)

log_powerlawf=dict

for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    name=temp[0].split(':')[2]
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name = temp[0][10:-1]
    time = float(temp1[1][1:-1])
    res = float(temp2[1][10:-4])
    res_th = float(temp3[1][10:-4])

    log_powerlawf[name]['time']=time
    log_powerlawf[name]['res'] = abs(int(res))
    log_powerlawf[name]['res_th'] = abs(int(res_th))






x_axis1=np.array([log_powerlawf[name]['p'] for name in log_powerlawf])
x_axis2=np.array([log_powerlawf[name]['Average Clustering Coefficient'] for name in log_powerlawf])
x_axis3=np.array([log_powerlawf[name]['Pearson Degree Correlation'] for name in log_powerlawf])
# x_axis4=np.array([log_powerlawf[name]['Average Shortest Path'] for name in log_powerlawf])
# x_axis5=np.array([log_powerlawf[name]['Diameter'] for name in log_powerlawf])
# x_axis6=np.array([log_powerlawf[name]['Eigen Vector Centrality']*log_powerlawf[name]['p'] for name in log_powerlawf])
x_axis7=np.array([log_powerlawf[name]['Edges']/log_powerlawf[name]['Nodes'] for name in log_powerlawf])


y_axis1=np.array([log_powerlawf[name]['res_th']/log_powerlawf[name]['res'] for name in log_powerlawf])



nd=len(y_axis1)


plotm=np.zeros([8,nd])

plotm[0,:]=x_axis1
plotm[1,:]=y_axis1
plotm[2,:]=x_axis2
plotm[3,:]=x_axis3
# plotm[4,:]=x_axis4
# plotm[5,:]=x_axis5
# plotm[6,:]=x_axis6
plotm[7,:]=x_axis7

plotms=plotm[:, plotm[0].argsort()]
plotms2=plotm[:, plotm[2].argsort()]
plotms3=plotm[:, plotm[3].argsort()]
plotms4=plotm[:, plotm[4].argsort()]
plotms5=plotm[:, plotm[5].argsort()]
plotms6=plotm[:, plotm[6].argsort()]
plotms7=plotm[:, plotm[7].argsort()]




plt.plot(plotms[0,:],plotms[1,:], marker='o', label=Type+'_N=200')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('Density')
plt.legend(loc='upper right')
plt.savefig(pathfig+Type+"_p_all2.png")
plt.show()