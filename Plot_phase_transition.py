import matplotlib.pyplot as plt
import json
import os
import numpy as np










# with open('maxind_rand_200_tot.log') as f:
#     Log=f.readlines()

with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_200_tot.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=200
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p

for p in np.linspace(0.01,0.1,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p


y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=200')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')
#plt.show()





# with open('maxind_rand_1000_tot.log') as f:
#     Log=f.readlines()

with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_1000_tot.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=1000
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p

for p in np.linspace(0.01,0.1,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p


y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=1000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')


# with open('maxind_rand_2000.log') as f:
#     Log=f.readlines()


with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_2000.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=2000
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p





y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=2000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')



# with open('maxind_rand_3000.log') as f:
#     Log=f.readlines()

with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_3000.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=3000
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p




y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=3000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')


# path='./res/plots/'
# os.chdir(path)

plt.savefig("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/phase_transit_random_lr_1e-4.pdf")
plt.show()



#lr=1e-4
n_vec=[200, 1000, 2000, 3000]
p_phase=[0.03, 0.009, 0.004, 0.003]



p_erdos=[np.log(item)/item for item in n_vec]

plt.plot(n_vec,p_phase, marker='o', label='Phase Transition Curve')
plt.ylabel('Transition Probability')
plt.xlabel('n')
plt.legend(loc='upper right')

plt.plot(n_vec,p_erdos, marker='o', label='ln(N)/N Curve')
plt.ylabel('Transition Probability')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/phase_transit_erdos_lr_1e-4.pdf")
plt.show()


#lr=1e-5


with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_200_sl_tot.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=200
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p

for p in np.linspace(0.01,0.1,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p


y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=200')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')
#plt.show()





# with open('maxind_rand_1000_tot.log') as f:
#     Log=f.readlines()

with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_1000_sl_tot.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=1000
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p

for p in np.linspace(0.01,0.1,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p


y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=1000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')


# with open('maxind_rand_2000.log') as f:
#     Log=f.readlines()


with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_2000_sl.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=2000
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p





y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=2000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')



# with open('maxind_rand_3000.log') as f:
#     Log=f.readlines()

with open('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/log/maxind_rand_3000_sl.log') as f:
    Log=f.readlines()

log_d={}
for lines in Log:
    temp=lines.split(',')
    temp1=temp[1].split(':')
    temp2 = temp[2].split(':')
    temp3 = temp[3].split(':')
    name=temp[0][10:-1]
    time=float(temp1[1][1:-1])
    res=float(temp2[1][10:-4])
    res_th=float(temp3[1][10:-4])
    log_d[name]={}
    log_d[name]['time']=time
    log_d[name]['res'] = abs(int(res))
    log_d[name]['res_th'] = abs(int(res_th))

n=3000
for p in np.linspace(0.001,0.01,10):
    name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
    log_d[name]['p']=p




y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])


x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])

nd=len(y_axis)


plotm=np.zeros([2,nd])

plotm[0,:]=x_axis
plotm[1,:]=y_axis

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=3000')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')




plt.savefig("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/phase_transit_random_lr_1e-5.pdf")
plt.show()



#lr=1e-5
n_vec=[200, 1000, 2000, 3000]
p_phase=[0.03, 0.008, 0.004, 0.003]



p_erdos=[np.log(item)/item for item in n_vec]

plt.plot(n_vec,p_phase, marker='o', label='Phase Transition Curve')
plt.ylabel('Transition Probability')
plt.xlabel('n')
plt.legend(loc='upper right')

plt.plot(n_vec,p_erdos, marker='o', label='ln(N)/N Curve')
plt.ylabel('Transition Probability')
plt.xlabel('Number of Nodes')
plt.legend(loc='upper right')
plt.savefig("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/phase_transit_erdos_lr_1e-5.pdf")
plt.show()




