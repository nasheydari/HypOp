import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.optimize import curve_fit



#synthetic

path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermincut_syn_2_m.log'
with open(path) as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(':')
    time=float(temp[4][1:-5])
    temp2 = temp[5][3:-11].split(',')
    res=int(temp2[1][1:])
    name=temp[2]
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    n = int(name.split('_')[1])
    m = int(name.split('_')[2][:-4])
    log_d[name]['n']=n
    log_d[name]['m']=m

path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermincut_syn_transfer_copy.log'

with open(path) as f:
    Log=f.readlines()

log_r={}
for lines in Log:
    temp = lines.split(':')
    time = float(temp[4][1:-5])
    temp2 = temp[5][3:-11].split(',')
    res = int(temp2[1][1:])
    name = temp[2]
    log_r[name] = {}
    log_r[name]['time'] = time
    log_r[name]['res'] = abs(int(res))
    n = int(name.split('_')[1])
    m = int(name.split('_')[2][:-4])
    log_r[name]['n'] = n
    log_r[name]['m'] = m


x_axis=np.array([log_d[name]['n'] for name in log_d])

y_axis=np.array([log_d[name]['res']/log_d[name]['m'] for name in log_d])
y_axis2=np.array([log_r[name]['res']/log_d[name]['m'] for name in log_d])


# y_axis_=np.array([log_d[name]['res_th']/log_d[name]['n'] for name in log_d])
# y_axis2_=np.array([log_g[name]['res_th']/log_d[name]['n'] for name in log_g])



y_axis_t=np.array([log_d[name]['time'] for name in log_d])
y_axis_t2=np.array([log_r[name]['time'] for name in log_d])
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


plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='x', label='Vanila Training')
plt.plot(plotms[0,:],plotms[2,:], marker='o', label='Transfer Learning')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig('./res/plots/Hypermincut_transfer.tiff')
plt.show()




def model_ex(x,a, b):
  return b*np.exp(a*x)

def model_l(x,a,b):
  return b*(x)+a





popt, pcov = curve_fit(model_l, plotms[0,:], plotms[3,:], p0=[0,0])
popt_r, pcov_r = curve_fit(model_l, plotms[0,:], plotms[4,:], p0=[0,0])

# popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])


a, b= popt
a_r, b_r = popt_r
# a_l2, b_l2= popt2

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)

y_model = model_l(x_model, a, b)


y_model2 = model_l(x_model, a_r, b_r)

# y_model3 = model_l(x_model, a_l2, b_l2)


plt.scatter(plotms[0,:], plotms[3,:], label='Vanila Training')
plt.scatter(plotms[0,:], plotms[4,:],  label='Transfer Learning')
plt.plot(x_model, y_model,linestyle='--' )
plt.plot(x_model, y_model2, linestyle='--')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermincut_transfer_time.tiff')
plt.show()




# plt.plot(plotms[0,:],plotms[5,:], marker='x', label='ADAM')
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
# plt.ylabel('Cut Size over the Number of Nodes')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.show()






