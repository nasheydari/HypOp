import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.optimize import curve_fit



#synthetic

path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/task_APS_SA_plot3.log'
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
    res=float(temp2[1][3:-2])
    res_th=float(temp3[1][3:-2])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))
    path="/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/data/hypergraph_data/APS/3years_Stanford/"+name
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    info = lines[0].split(' ')
    n = int(info[0])
    m = int(info[1])
    log_d[name]['n']=n
    log_d[name]['m']=m

path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/task_APS_vec_plot3.log'

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
    res=float(temp2[1][3:-2])
    res_th=float(temp3[1][3:-2])
    log_r[name]={}
    log_r[name]['time']=time
    log_r[name]['res'] = abs(int(res))
    log_r[name]['res_th'] = abs(int(res_th))


x_axis=np.array([log_d[name]['n'] for name in log_d])

y_axis=np.array([log_d[name]['res']/log_d[name]['n'] for name in log_d])
y_axis2=np.array([log_r[name]['res']/log_d[name]['n'] for name in log_r])


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
plotm[3,:]=y_axis_t2
plotm[4,:]=y_axis_t
# plotm[5,:]=y_axis3
# plotm[6,:]=y_axis_t3
# plotm[7,:]=y_axis_
# plotm[8,:]=y_axis2_


plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[2,:], marker='x', label='HypOp')
plt.plot(plotms[0,:],plotms[1,:], marker='o', label='SA')
plt.ylabel('Average Unsatisfied Constraints per Agent')
plt.xlabel('Number of Agents')
plt.legend(loc='upper left')
plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/Task3.png')
plt.show()




def model_ex(x,a, b):
  return b*np.exp(a*x)

def model_l(x,a,b):
  return b*(x)+a


popt_r, pcov_r = curve_fit(model_ex, plotms[0,:], plotms[3,:], p0=[0,0])

a_ex_r, b_ex_r= popt_r

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)
y_model = model_ex(x_model, a_ex_r, b_ex_r)

popt_d, pcov_d = curve_fit(model_ex, plotms[0,:], plotms[4,:], p0=[0,0])

# popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])

a_ex_d, b_ex_d= popt_d

# a_l2, b_l2= popt2

y_model2 = model_ex(x_model, a_ex_d, b_ex_d)

# y_model3 = model_l(x_model, a_l2, b_l2)


plt.scatter(plotms[0,:], plotms[4,:], label='SA')
plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
plt.plot(x_model, y_model2, color='r', label='Exponential Curve Fit for SA')
plt.plot(x_model, y_model, color='y', label='EXponential Curve Fit for HypOp')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='lower left')
# plt.savefig('./res/plots/Hypermaxcut_syn_time.png')
plt.show()




# plt.plot(plotms[0,:],plotms[5,:], marker='x', label='ADAM')
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
# plt.ylabel('Cut Size over the Number of Nodes')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.show()





