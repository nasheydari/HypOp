import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.optimize import curve_fit



#synthetic
path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_syn_new_rev1_2_plot5.log'
# path='../log/Hypermaxcut_syn_new_rev1_integrated.log'
with open(path) as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    temp3 = temp[6][1:-7].split(',')
    res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total= [time_mlist[i]+time_tlist[i] for i in range(len(time_tlist))]
    temp2=temp2[:-1]
    name=temp[2]
    log_d[name]={}
    log_d[name]['time_train']=time_tlist
    log_d[name]['time_map'] = time_mlist
    log_d[name]['time_total'] = time_total
    log_d[name]['res'] = reslist
    log_d[name]['res_th'] = res_thlist
    path="/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/data/hypergraph_data/synthetic/new/all/"+name
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    info = lines[0].split(' ')
    n = int(info[0])
    m = int(info[1])
    log_d[name]['n']=n
    log_d[name]['m']=m



path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_syn_new_Rand_l2_rev1_s_2_plot5.log'

with open(path) as f:
    Log=f.readlines()

log_r={}
for lines in Log:
    temp = lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    # temp3 = temp[6][1:-7].split(',')
    # res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
    temp2 = temp2[:-1]
    name = temp[2]
    log_r[name] = {}
    log_r[name]['time_train'] = time_tlist
    log_r[name]['time_map'] = time_mlist
    log_r[name]['time_total'] = time_total
    log_r[name]['res'] = reslist


path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_syn_new_rev1_2_GD5.log'

with open(path) as f:
    Log=f.readlines()

log_b={}
for lines in Log:
    temp = lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    # temp3 = temp[6][1:-7].split(',')
    # res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
    temp2 = temp2[:-1]
    # name = temp[2][:-14]+'.txt' #bipartite
    name = temp[2]
    log_b[name] = {}
    log_b[name]['time_train'] = time_tlist
    log_b[name]['time_map'] = time_mlist
    log_b[name]['time_total'] = time_total
    log_b[name]['res'] = reslist


path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_syn_new_bipartite_rev1_copy_5.log'

with open(path) as f:
    Log=f.readlines()

log_bi={}
for lines in Log:
    temp = lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    # temp3 = temp[6][1:-7].split(',')
    # res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
    temp2 = temp2[:-1]
    name = temp[2][:-14]+'.txt' #bipartite
    log_bi[name] = {}
    log_bi[name]['time_train'] = time_tlist
    log_bi[name]['time_map'] = time_mlist
    log_bi[name]['time_total'] = time_total
    log_bi[name]['res'] = reslist


path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_syn_att.log'

with open(path) as f:
    Log=f.readlines()

log_a={}
for lines in Log:
    temp = lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    # temp3 = temp[6][1:-7].split(',')
    # res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
    temp2 = temp2[:-1]
    name = temp[2]
    log_a[name] = {}
    log_a[name]['time_train'] = time_tlist
    log_a[name]['time_map'] = time_mlist
    log_a[name]['time_total'] = time_total
    log_a[name]['res'] = reslist

name_att=[name for name in log_d if name!='Hyp_8901_13058.txt' and name!='Hyp_6281_9325.txt']

x_axis=np.array([log_d[name]['n'] for name in log_d])

y_axis=np.array([np.average(log_d[name]['res'])/log_d[name]['m'] for name in log_d])
y_axis_error=np.array([np.std(log_d[name]['res'])/log_d[name]['m'] for name in log_d])

y_axis2=np.array([np.average(log_r[name]['res'])/log_d[name]['m'] for name in log_d])
y_axis2_error=np.array([np.std(log_r[name]['res'])/log_d[name]['m'] for name in log_d])

y_axisb=np.array([np.average(log_b[name]['res'])/log_d[name]['m'] for name in log_d])
y_axisb_error=np.array([np.std(log_b[name]['res'])/log_d[name]['m'] for name in log_d])

y_axisbi=np.array([np.average(log_bi[name]['res'])/log_d[name]['m'] for name in log_d])
y_axisbi_error=np.array([np.std(log_bi[name]['res'])/log_d[name]['m'] for name in log_d])



# y_axis_=np.array([log_d[name]['res_th']/log_d[name]['n'] for name in log_d])
# y_axis2_=np.array([log_g[name]['res_th']/log_d[name]['n'] for name in log_g])



y_axis_t=np.array([np.average(log_d[name]['time_total']) for name in log_d])
y_axis_t_error=np.array([np.std(log_d[name]['time_total']) for name in log_d])
y_axis_t2=np.array([np.average(log_r[name]['time_total']) for name in log_d])
y_axis_t2_error=np.array([np.std(log_r[name]['time_total']) for name in log_d])

y_axis_tb=np.array([np.average(log_b[name]['time_total']) for name in log_d])
y_axis_tb_error=np.array([np.std(log_b[name]['time_total']) for name in log_d])


y_axis_tbi=np.array([np.average(log_bi[name]['time_total']) for name in log_d])
y_axis_tbi_error=np.array([np.std(log_bi[name]['time_total']) for name in log_d])

# y_axis_t3=np.array([log_g[name]['time'] for name in log_g])



x_axis_s=np.array([log_d[name]['n'] for name in name_att])

y_axis_s=np.array([np.average(log_d[name]['res'])/log_d[name]['m'] for name in name_att])
y_axis_s_error=np.array([np.std(log_d[name]['res'])/log_d[name]['m'] for name in name_att])

y_axisa=np.array([np.average(log_a[name]['res'])/log_d[name]['m'] for name in name_att])
y_axisa_error=np.array([np.std(log_a[name]['res'])/log_d[name]['m'] for name in name_att])

y_axis_t_s=np.array([np.average(log_d[name]['time_total']) for name in name_att])
y_axis_t_s_error=np.array([np.std(log_d[name]['time_total']) for name in name_att])

y_axis_ta=np.array([np.average(log_a[name]['time_total']) for name in name_att])
y_axis_ta_error=np.array([np.std(log_a[name]['time_total']) for name in name_att])

plotma=np.zeros([6,len(name_att)])

plotma[0,:]=x_axis_s
plotma[1,:]=y_axis_s
plotma[2,:]=y_axisa
plotma[3,:]=y_axis_t_s
plotma[4,:]=y_axis_ta
plotma[5,:]=y_axis_s_error

plotmas=plotma[:, plotma[0].argsort()]





nd=len(y_axis)


plotm=np.zeros([15,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis
plotm[2,:]=y_axis2
plotm[3,:]=y_axis_t
plotm[4,:]=y_axis_t2
plotm[5,:]=y_axis_error
plotm[6,:]=y_axis2_error
plotm[7,:]=y_axis_t_error
plotm[8,:]=y_axis_t2_error
plotm[9,:]=y_axisb
plotm[10,:]=y_axisb_error
plotm[11,:]=y_axis_tb
plotm[12,:]=y_axis_tb_error
plotm[13,:]=y_axisbi
plotm[14,:]=y_axis_tbi


plotms=plotm[:, plotm[0].argsort()]

y_lower=plotms[1,:]-plotms[5,:]
y_upper=plotms[1,:]+plotms[5,:]
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='HypOp')
# plt.plot(plotms[0,:],plotms[2,:], marker='x', label='SA')
plt.errorbar(plotms[0,:],plotms[1,:], marker='.', label='HypOp')
plt.errorbar(plotms[0,:],plotms[2,:], marker='.', label='SA')
plt.errorbar(plotms[0,:],plotms[9,:], marker='.', label='ADAM')
# plt.errorbar(plotms[0,:],plotms[9,:], yerr = plotms[10,:], label='Bipartite')
plt.fill_between(plotms[0,:], y_lower, y_upper, color='gray', alpha=0.4, label='HypOp Error Region')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='lower right')
plt.ylim(0.945,0.98)
plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/Hypermaxcut_syn_rev1_plot5_2.pdf')
plt.show()




def model_ex(x,a, b):
  return b*np.exp(a*x)

def model_l(x,a,b):
  return b*(x)+a

def model_quad(x, a, b):
    y=[b*xi**2+a for xi in x]
    return y


popt, pcov = curve_fit(model_ex, plotms[0,:], plotms[4,:], p0=[0,0])

poptq2, pcovq2 = curve_fit(model_quad, plotms[0,:], plotms[4,:], p0=[0,0])

poptq3, pcovq3 = curve_fit(model_quad, plotms[0,:], plotms[11,:], p0=[0,0])

poptqb, pcovqb = curve_fit(model_quad, plotms[0,:], plotms[14,:], p0=[0,0])

a_ex, b_ex= popt
a_q2, b_q2= poptq2
a_q3, b_q3= poptq3
a_qb, b_qb= poptqb

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)
x_model_s = np.linspace(min(plotmas[0,:]), max(plotmas[0,:]), 100)

y_model = model_ex(x_model, a_ex, b_ex)
y_modelq2 = model_quad(x_model, a_q2, b_q2)
y_modelq3 = model_quad(x_model, a_q3, b_q3)
y_modelqb = model_quad(x_model, a_qb, b_qb)


popt, pcov = curve_fit(model_l, plotms[0,:], plotms[3,:], p0=[0,0])

popt_s, pcov_s = curve_fit(model_l, plotmas[0,:], plotmas[3,:], p0=[0,0])

poptb, pcovb = curve_fit(model_l, plotms[0,:], plotms[11,:], p0=[0,0])

poptbb, pcovbb = curve_fit(model_l, plotms[0,:], plotms[14,:], p0=[0,0])

popta, pcova = curve_fit(model_ex, plotmas[0,:], plotmas[4,:], p0=[0,0])

# popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])

a_l, b_l= popt

a_l_s, b_l_s= popt_s


a_lb, b_lb= poptb
a_lbb, b_lbb= poptbb

a_exa, b_exa= popta

y_model2 = model_l(x_model, a_l, b_l)
y_model_s = model_l(x_model_s, a_l_s, b_l_s)

y_modelb = model_l(x_model, a_lb, b_lb)
y_modelbb = model_l(x_model, a_lbb, b_lbb)


y_modela = model_ex(x_model_s, a_exa, b_exa)




# plt.scatter(plotms[0,:], plotms[3,:], color='g', label='HypOp')
# plt.scatter(plotms[0,:], plotms[4,:], label='SA')
plt.scatter(plotms[0,:], plotms[3,:],  label='HypOp')
plt.plot(x_model, y_model2, color='b', linestyle='--', label='Linear Curve Fit')
plt.scatter(plotms[0,:], plotms[4,:],   label='SA')
plt.plot(x_model, y_modelq2, color='r', linestyle='--', label='Quadratic Curve Fit')
plt.scatter(plotms[0,:], plotms[11,:],  label='ADAM')
plt.plot(x_model, y_modelq3, color='g', linestyle='--', label='Quadratic Curve Fit')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_syn_time_rev1_plot5_2.pdf')
plt.show()



##bipartite
plt.errorbar(plotms[0,:],plotms[1,:], marker='.', label='HypOp')
plt.errorbar(plotms[0,:],plotms[13,:], marker='.', label='Bipartite GNN')
# plt.errorbar(plotms[0,:],plotms[9,:], yerr = plotms[10,:], label='Bipartite')
plt.fill_between(plotms[0,:], y_lower, y_upper, color='gray', alpha=0.4, label='HypOp Error Region')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='lower right')
plt.ylim(0.945,0.98)
plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/Hypermaxcut_syn_rev1_plot5_bipartite.jpg')
plt.show()

plt.scatter(plotms[0,:], plotms[3,:],  label='HypOp')
plt.plot(x_model, y_model2, color='b', linestyle='--', label='Linear Curve Fit')
plt.scatter(plotms[0,:], plotms[14,:],  label='Bipartite GNN')
plt.plot(x_model, y_modelbb, color='r', linestyle='--', label='Linear Curve Fit')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_syn_time_rev1_plot5__bipartite.jpg')
plt.show()



##attention

y_lower_s=plotmas[1,:]-plotmas[5,:]
y_upper_s=plotmas[1,:]+plotmas[5,:]
plt.errorbar(plotmas[0,:],plotmas[1,:], marker='.', label='HypOp')
plt.errorbar(plotmas[0,:],plotmas[2,:], marker='.', label='Hypergraph Attention')
# plt.errorbar(plotms[0,:],plotms[9,:], yerr = plotms[10,:], label='Bipartite')
plt.fill_between(plotmas[0,:], y_lower_s, y_upper_s, color='gray', alpha=0.4, label='HypOp Error Region')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='lower right')

plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/Hypermaxcut_syn_rev1_plot5_attention.pdf')
plt.show()



plt.scatter(plotmas[0,:], plotmas[3,:],  label='Hypergraph Convolution (HypOp)')
plt.plot(x_model_s, y_model2, color='b', linestyle='--', label='Linear Curve Fit')
plt.scatter(plotmas[0,:], plotmas[4,:],  label='Hypergraph Attention')
plt.plot(x_model_s, y_modela, color='r', linestyle='--', label='Exponential Curve Fit')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_syn_time_rev1_plot5_attention.pdf')
plt.show()













#APS
path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_APS_3Y_f2_rev1_10.log'
with open(path) as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp = lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    temp3 = temp[6][1:-7].split(',')
    res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
    temp2 = temp2[:-1]
    name = temp[2]
    log_d[name] = {}
    log_d[name]['time_train'] = time_tlist
    log_d[name]['time_map'] = time_mlist
    log_d[name]['time_total'] = time_total
    log_d[name]['res'] = reslist
    log_d[name]['res_th'] = res_thlist
    path = "/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/data/hypergraph_data/APS/3years_Stanford/"+name
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    info = lines[0].split(' ')
    n = int(info[0])
    m = int(info[1])
    log_d[name]['n'] = n
    log_d[name]['m'] = m



path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_APS_3y_Rand_rev1_10.log'

with open(path) as f:
    Log=f.readlines()

log_r={}
for lines in Log:
    temp = lines.split(':')
    temp2 = temp[5][1:-9].split(',')
    reslist = [abs(int(temp2[i][2:-1])) for i in range(len(temp2))]
    # temp3 = temp[6][1:-7].split(',')
    # res_thlist = [abs(int(temp3[i][2:-1])) for i in range(len(temp3))]
    temp4 = temp[8][2:-16].split(', ')
    time_tlist = [float(temp4[i]) for i in range(len(temp4))]
    temp5 = temp[9][2:-5].split(', ')
    time_mlist = [float(temp5[i]) for i in range(len(temp5))]
    time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
    temp2 = temp2[:-1]
    name = temp[2]
    log_r[name] = {}
    log_r[name]['time_train'] = time_tlist
    log_r[name]['time_map'] = time_mlist
    log_r[name]['time_total'] = time_total
    log_r[name]['res'] = reslist

path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_APS_3y_GD_rev1_s2.log'
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


path='/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/Hypermaxcut_APS_3y_GD_rev1.log'
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

y_axis=np.array([np.average(log_d[name]['res'])/log_d[name]['m'] for name in log_d])
y_axis_error=np.array([np.std(log_d[name]['res'])/log_d[name]['m'] for name in log_d])


y_axis2=np.array([np.average(log_r[name]['res'])/log_d[name]['m'] for name in log_d])
y_axis2_error=np.array([np.std(log_r[name]['res'])/log_d[name]['m'] for name in log_d])


y_axis3=np.array([np.average(log_g[name]['res'])/log_d[name]['m'] for name in log_d])
y_axis3_error=np.array([np.std(log_g[name]['res'])/log_d[name]['m'] for name in log_d])


y_axis_=np.array([np.average(log_d[name]['res_th'])/log_d[name]['m'] for name in log_d])
y_axis2_=np.array([np.average(log_g[name]['res_th'])/log_d[name]['m'] for name in log_d])
y_axis3_=np.array([np.average(log_g2[name]['res_th'])/log_d[name]['m'] for name in log_d])



y_axis_t=np.array([np.average(log_d[name]['time_train']) for name in log_d])
y_axis_t_error=np.array([np.std(log_d[name]['time_train']) for name in log_d])

y_axis_t2=np.array([np.average(log_r[name]['time_map']) for name in log_d])
y_axis_t2_error=np.array([np.std(log_r[name]['time_map']) for name in log_d])

y_axis_t3=np.array([np.average(log_g[name]['time']) for name in log_d])
y_axis_t3_error=np.array([np.std(log_g[name]['time']) for name in log_d])
y_axis_t4=np.array([np.average(log_g2[name]['time']) for name in log_d])


nd=len(y_axis)


plotm=np.zeros([17,nd])

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
plotm[11,:]=y_axis_error
plotm[12,:]=y_axis2_error
plotm[13,:]=y_axis3_error
plotm[14,:]=y_axis_t_error
plotm[15,:]=y_axis_t2_error
plotm[16,:]=y_axis_t3_error

plotms=plotm[:, plotm[0].argsort()]


y_upper=plotms[1,:]+plotms[11,:]
y_lower=plotms[1,:]-plotms[11,:]
y_upper2=plotms[2,:]+plotms[12,:]
y_lower2=plotms[2,:]-plotms[12,:]
y_upper3=plotms[5,:]+plotms[13,:]
y_lower3=plotms[5,:]-plotms[13,:]

plt.errorbar(plotms[0,:],plotms[1,:],  marker='.', label='HypOp')
plt.errorbar(plotms[0,:],plotms[2,:], marker='.', label='SA')
plt.errorbar(plotms[0,:],plotms[5,:], marker='.', label='ADAM')
plt.fill_between(plotms[0,:], y_lower, y_upper, color='gray', alpha=0.4, label='HypOp Error Region')
# plt.fill_between(plotms[0,:], y_lower2, y_upper2, color='gray', alpha=0.4, label='Error Region')
plt.ylabel('Cut Size over the Number of Hyperedges')
plt.xlabel('Number of Nodes')
plt.legend(loc='lower right')
plt.savefig('./res/plots/Hypermaxcut_APS_rev1_3.pdf')
plt.show()

# plt.plot(plotms[0,:],plotms[4,:], marker='X', label='SA')
# plt.plot(plotms[0,:],plotms[3,:], marker='o', label='HypOp')
# plt.ylabel('Runtime (s)')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.show()


#
# plt.errorbar(plotms[0,:],plotms[7,:],   marker='x', label='HypOp')
# plt.errorbar(plotms[0,:],plotms[8,:],   marker='o', label='ADAM')
# plt.ylabel('Cut Size over the Number of Hyperedges')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='lower right')
# plt.savefig('./res/plots/Hypermaxcut_APS_ADAM_l2_rev1.png')
# plt.show()
#
# plt.plot(plotms[0,:],plotms[7,:], marker='x', label='HypOp')
# plt.plot(plotms[0,:],plotms[9,:], marker='o', label='ADAM')
# plt.ylabel('Cut Size over the Number of Hyperedges')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='lower right')
# plt.savefig('./res/plots/Hypermaxcut_APS_ADAM_rev1.png')
# plt.show()
#


popt, pcov = curve_fit(model_ex, plotms[0,:], plotms[4,:], p0=[0,0])
poptq, pcovq = curve_fit(model_quad, plotms[0,:], plotms[4,:], p0=[0,0])




a_ex, b_ex= popt
a_q, b_q= poptq

x_model = np.linspace(min(plotms[0,:]), max(plotms[0,:]), 100)
y_model = model_ex(x_model, a_ex, b_ex)
y_modelq = model_quad(x_model, a_q, b_q)

popt, pcov = curve_fit(model_l, plotms[0,:], plotms[3,:], p0=[0,0])

popt2, pcov2 = curve_fit(model_l, plotms[0,:], plotms[6,:], p0=[0,0])

popt3, pcov3 = curve_fit(model_l, plotms[0,:], plotms[10,:], p0=[0,0])


a_l, b_l= popt

a_l2, b_l2= popt2

a_l3, b_l3= popt3

y_model2 = model_l(x_model, a_l, b_l)

y_model3 = model_l(x_model, a_l2, b_l2)

y_model4 = model_l(x_model, a_l3, b_l3)

plt.scatter(plotms[0,:], plotms[3,:],   label='HypOp')
plt.plot(x_model, y_model2, color='b', linestyle='--', label='Linear Curve Fit for HypOp')
plt.scatter(plotms[0,:], plotms[4,:],  label='SA')
plt.plot(x_model, y_modelq, color='r',linestyle='--', label='Quadratic Curve Fit for SA')
plt.scatter(plotms[0,:], plotms[6,:],  label='ADAM')
plt.plot(x_model, y_model3, color='g',linestyle='--', label='Linear Curve Fit for ADAM')
plt.ylabel('Run time (s)')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper left')
plt.savefig('./res/plots/Hypermaxcut_APS_time_fit_rev1_3.pdf')
plt.show()







