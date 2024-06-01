import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.optimize import curve_fit



#synthetic

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/maxind_regular_d5_l.log'
with open(path) as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0].split(':')[2]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][11:-4])
    res_th=-1*float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))
    n = int(name[6:-6])
    d=3
    log_d[name]['n']=n
    log_d[name]['d']=d

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/maxind_regular_d5_GNN.log'

with open(path) as f:
    Log=f.readlines()

log_r={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0].split(':')[2]
    time=float(temp1[1][1:-1])
    res = float(temp2[1][11:-4])
    res_th = -1*float(temp3[1][10:-4])
    log_r[name]={}
    log_r[name]['time']=time
    log_r[name]['res'] = abs(int(res))
    log_r[name]['res_th'] = abs(int(res_th))


x_axis=np.array([log_d[name]['n'] for name in log_d])

y_axis=np.array([log_d[name]['res']/log_d[name]['n'] for name in log_d])
y_axis2=np.array([log_r[name]['res_th']/log_d[name]['n'] for name in log_r])


# y_axis_=np.array([log_d[name]['res_th']/log_d[name]['n'] for name in log_d])
# y_axis2_=np.array([log_g[name]['res_th']/log_d[name]['n'] for name in log_g])



y_axis_t=np.array([log_d[name]['time'] for name in log_d])
y_axis_t2=np.array([log_r[name]['time'] for name in log_r])
# y_axis_t3=np.array([log_g[name]['time'] for name in log_g])


nd=len(y_axis)


plotm=np.zeros([10,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis
plotm[2,:]=y_axis2
plotm[3,:]=y_axis_t
plotm[4,:]=y_axis_t2
# plotm[5,:]=y_axis3
# plotm[6,:]=y_axis_t3
# plotm[7,:]=y_axis_
# plotm[8,:]=y_axis2_

plotm[5,:]= [ 0.00257264, 0.00217485, 0.00204782, 0.00150146, 0.00125629,0.00114555,0.00080135, 0.00078274]
plotms=plotm[:, plotm[0].argsort()]


y_lower=plotms[1,:]-plotms[5,:]
y_upper=plotms[1,:]+plotms[5,:]

plt.errorbar(plotms[0,:],plotms[1,:], marker='.', label='HypOp')
plt.errorbar(plotms[0,:],plotms[2,:], marker='.', label='PI-GNN')
# plt.errorbar(plotms[0,:],plotms[9,:], yerr = plotms[10,:], label='Bipartite')
plt.fill_between(plotms[0,:], y_lower, y_upper, color='gray', alpha=0.4, label='HypOp Error Region')
plt.ylabel('MIS Size over the Number of Nodes')
plt.xlabel('Number of Nodes')
plt.legend(loc='lower left')
plt.ylim(0.31,0.37)
plt.savefig('./res/plots/Maxind_regular_d5_rev1_5.pdf')
plt.show()






def model_ex(x,a, b):
  return b*np.exp(a*x)

def model_l(x,a,b):
  return b*(x)+a


popt, pcov = curve_fit(model_ex, plotms[0,:], plotms[4,:], p0=[0,0])

a_r, b_r= popt

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)
y_model = model_ex(x_model, a_r, b_r)

popt, pcov = curve_fit(model_ex, plotms[0,:], plotms[3,:], p0=[0,0])

# popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])

a_l, b_l= popt

# a_l2, b_l2= popt2

y_model2 = model_ex(x_model, a_l, b_l)

# y_model3 = model_l(x_model, a_l2, b_l2)


plt.scatter(plotms[0,:], plotms[4,:], label='PI-GNN')
plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
plt.plot(x_model, y_model, color='r')
plt.plot(x_model, y_model2, color='y')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Maxind_regular_d5_time_rev1.pdf')
plt.show()











