import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.optimize import curve_fit



#synthetic

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/maxcut_stanford_p1_f.log'
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
    path="./data/maxcut_data/stanford_data_p1/"+name
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    info = lines[0].split(' ')
    n = int(info[0])
    m = int(info[1])
    log_d[name]['n']=n
    log_d[name]['m']=m

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/maxcut_stanford_p1_Rand_l2.log'

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

y_axis=np.array([log_d[name]['res']/log_d[name]['m'] for name in log_d])
y_axis2=np.array([log_r[name]['res']/log_d[name]['m'] for name in log_r])


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


plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[2,:], marker='x', label='SA')
plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
plt.ylabel('Cut Size over the Number of Edges')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig('./res/plots/maxcut_stanford_p1.png')
plt.show()




def model_ex(x,a, b):
  return b*np.exp(a*x)

def model_l(x,a,b):
  return b*(x)+a


popt, pcov = curve_fit(model_ex, plotms[0,:], plotms[4,:], p0=[0,0])

a_ex, b_ex= popt

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)
y_model = model_ex(x_model, a_ex, b_ex)

popt, pcov = curve_fit(model_l, plotms[0,:], plotms[3,:], p0=[0,0])

# popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])

a_l, b_l= popt

# a_l2, b_l2= popt2

y_model2 = model_l(x_model, a_l, b_l)

# y_model3 = model_l(x_model, a_l2, b_l2)


plt.scatter(plotms[0,:], plotms[4,:], label='SA')
plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
plt.plot(x_model, y_model, color='r', label='Exponential Curve Fit')
plt.plot(x_model, y_model2, color='y', label='Linear Curve Fit')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/maxcut_stanford_p1_time.png')
plt.show()




# plt.plot(plotms[0,:],plotms[5,:], marker='x', label='ADAM')
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
# plt.ylabel('Cut Size over the Number of Nodes')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.show()


Best_known={}
Best_known['G1.txt']={}
Best_known['G1.txt']['res']=11624





#APS
path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/maxcut_APS_3Y_f2.log'
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
    path="./data/hypergraph_data/APS/3years_Stanford/"+name
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    info = lines[0].split(' ')
    n = int(info[0])
    m = int(info[1])
    log_d[name]['n']=n
    log_d[name]['m']=m

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/Hypermaxcut_APS_3Y_Rand_l.log'

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

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/Hypermaxcut_APS_3Y_GD3.log'
with open(path) as f:
    Log=f.readlines()

log_g={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0].split(':')[2]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][3:-2])
    res_th=float(temp3[1][3:-2])
    log_g[name]={}
    log_g[name]['time']=time
    log_g[name]['res'] = abs(int(res))
    log_g[name]['res_th'] = abs(int(res_th))


path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/Hypermaxcut_APS_3Y_GD.log'
with open(path) as f:
    Log=f.readlines()

log_g2={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0].split(':')[2]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][3:-2])
    res_th=float(temp3[1][3:-2])
    log_g2[name]={}
    log_g2[name]['time']=time
    log_g2[name]['res'] = abs(int(res))
    log_g2[name]['res_th'] = abs(int(res_th))

x_axis=np.array([log_d[name]['n'] for name in log_d])

y_axis=np.array([log_d[name]['res']/log_d[name]['m'] for name in log_d])
y_axis2=np.array([log_r[name]['res']/log_d[name]['m'] for name in log_r])
y_axis3=np.array([log_g[name]['res']/log_d[name]['m'] for name in log_g])


y_axis_=np.array([log_d[name]['res_th']/log_d[name]['m'] for name in log_d])
y_axis2_=np.array([log_g[name]['res_th']/log_d[name]['m'] for name in log_g])
y_axis3_=np.array([log_g2[name]['res_th']/log_d[name]['m'] for name in log_g2])



y_axis_t=np.array([log_d[name]['time'] for name in log_d])
y_axis_t2=np.array([log_r[name]['time'] for name in log_r])
y_axis_t3=np.array([log_g[name]['time'] for name in log_g])
y_axis_t4=np.array([log_g2[name]['time'] for name in log_g2])


nd=len(y_axis)


plotm=np.zeros([11,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis
plotm[2,:]=y_axis2
plotm[3,:]=y_axis_t
plotm[4,:]=y_axis_t2
plotm[5,:]=y_axis3
plotm[6,:]=y_axis_t3
plotm[7,:]=y_axis_
plotm[8,:]=y_axis2_
plotm[9,:]=y_axis3_
plotm[10,:]=y_axis_t4


plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[2,:], marker='x', label='SA')
plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig('./res/plots/Hypermaxcut_APS.png')
plt.show()

# plt.plot(plotms[0,:],plotms[4,:], marker='X', label='SA')
# plt.plot(plotms[0,:],plotms[3,:], marker='o', label='HypOp')
# plt.ylabel('Runtime (s)')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.show()

plt.plot(plotms[0,:],plotms[7,:], marker='x', label='HypOp')
plt.plot(plotms[0,:],plotms[8,:], marker='o', label='ADAM')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig('./res/plots/Hypermaxcut_APS_ADAM_l2.png')
plt.show()

plt.plot(plotms[0,:],plotms[7,:], marker='x', label='HypOp')
plt.plot(plotms[0,:],plotms[9,:], marker='o', label='ADAM')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig('./res/plots/Hypermaxcut_APS_ADAM.png')
plt.show()

def model_ex(x,a, b):
  return b*np.exp(a*x)

def model_l(x,a,b):
  return b*(x)+a


popt, pcov = curve_fit(model_ex, plotms[0,:], plotms[4,:], p0=[0,0])

a_ex, b_ex= popt

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)
y_model = model_ex(x_model, a_ex, b_ex)

popt, pcov = curve_fit(model_l, plotms[0,:], plotms[3,:], p0=[0,0])

popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])

popt3, pcov3 = curve_fit(model_l, plotms[0,:], plotms[10,:], p0=[0,0])


a_l, b_l= popt

a_l2, b_l2= popt2

a_l3, b_l3= popt3

y_model2 = model_l(x_model, a_l, b_l)

y_model3 = model_l(x_model, a_l2, b_l2)

y_model4 = model_l(x_model, a_l3, b_l3)

plt.scatter(plotms[0,:], plotms[4,:], label='SA')
plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
plt.plot(x_model, y_model, color='r', label='Exponential Curve Fit for SA')
plt.plot(x_model, y_model2, color='y', label='Linear Curve Fit for HypOp')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_APS_time_fit.png')
plt.show()




# plt.plot(plotms[0,:],plotms[5,:], marker='x', label='ADAM')
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
# plt.ylabel('Cut Size over the Number of Hyperedges')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.show()


plt.scatter(plotms[0,:], plotms[6,:], label='ADAM')
plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
plt.plot(x_model, y_model2, color='y', label='Linear Curve Fit for HypOp')
plt.plot(x_model, y_model3, color='r', label='Linear Curve Fit for ADAM')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_APS_ADAM_time_l.png')
plt.show()


plt.scatter(plotms[0,:], plotms[10,:], label='ADAM')
plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
plt.plot(x_model, y_model2, color='y', label='Linear Curve Fit for HypOp')
plt.plot(x_model, y_model4, color='r', label='Linear Curve Fit for ADAM')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_APS_ADAM_time.png')
plt.show()