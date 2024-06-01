import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pickle

path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/'



path2="/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/dist/good/"
path3="/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/dist/bad/"


i=0
for file_name in os.listdir(path2):
    i+=1
    if not file_name.startswith('.'):
        with open(path2+file_name, "rb") as fp:
            b = pickle.load(fp)
        plt.plot([1,2,3,4], b[-1], color=[0.3,i/11,1])
plt.xticks([1,2,3,4])
plt.ylabel('Max node embedding distance')
plt.xlabel('Main operations in different GNN layers')
plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/layers_all_good.pdf')
plt.show()

i=0
for file_name in os.listdir(path3):
    i+=1
    if not file_name.startswith('.'):
        with open(path3+file_name, "rb") as fp:
            b = pickle.load(fp)
        plt.plot(b[-1], color=[i/11,0.5,1])
plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/layers_all_bad.pdf')
plt.show()

# for file_name in os.listdir(path2):
#     if not file_name.startswith('.'):
#         with open(path2+file_name, "rb") as fp:
#             b = pickle.load(fp)
#         plt.plot(b[-1])
#         plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/'+file_name[:-4]+'layers.pdf')
#         plt.show()
        # plt.plot(range(len(b)),[item[3] for item in b])
        # plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/' + file_name[:-4] +'lastlayer.pdf')
        # plt.show()
        # plt.plot(range(len(b)),[item[0] for item in b])
        # plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/' + file_name[:-4] +'firstlayer.pdf')
        # plt.show()
# with open('maxind_rand_200_tot.log') as f:
#     Log=f.readlines()



with open(path+'maxind_rand_200_l_smooth.log') as f:
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

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Sparsified_Ps=0.8')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')


with open(path+'maxind_rand_200_l_sparse7.log') as f:
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
# for p in np.linspace(0.001,0.01,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p

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

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Sparsified_Ps=0.7')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')
# plt.show()


with open(path+'maxind_rand_200_l_sparse5.log') as f:
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
# for p in np.linspace(0.001,0.01,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p

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

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Sparsified_Ps=0.5')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')



with open(path+'maxind_rand_200_l.log') as f:
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
# for p in np.linspace(0.001,0.01,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p

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

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Original')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('ln(p)')
plt.legend(loc='upper right')
plt.savefig("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12-2023/res/plots/maxind_rand_sparse2.pdf")
plt.show()


#
#
# # with open('maxind_rand_1000_tot.log') as f:
# #     Log=f.readlines()
#
# with open('maxind_rand_1000_tot.log') as f:
#     Log=f.readlines()
#
# log_d={}
# for lines in Log:
#     temp=lines.split(',')
#     temp1=temp[1].split(':')
#     temp2 = temp[2].split(':')
#     temp3 = temp[3].split(':')
#     name=temp[0][10:-1]
#     time=float(temp1[1][1:-1])
#     res=float(temp2[1][10:-4])
#     res_th=float(temp3[1][10:-4])
#     log_d[name]={}
#     log_d[name]['time']=time
#     log_d[name]['res'] = abs(int(res))
#     log_d[name]['res_th'] = abs(int(res_th))
#
# n=1000
# for p in np.linspace(0.001,0.01,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p
#
# for p in np.linspace(0.01,0.1,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p
#
#
# y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])
#
#
# x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])
#
# nd=len(y_axis)
#
#
# plotm=np.zeros([2,nd])
#
# plotm[0,:]=x_axis
# plotm[1,:]=y_axis
#
# plotms=plotm[:, plotm[0].argsort()]
#
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=1000')
# plt.ylabel('Performance of HypOp for Maxind on random graphs')
# plt.xlabel('ln(p)')
# plt.legend(loc='upper right')
#
#
# # with open('maxind_rand_2000.log') as f:
# #     Log=f.readlines()
#
#
# with open('maxind_rand_2000.log') as f:
#     Log=f.readlines()
#
# log_d={}
# for lines in Log:
#     temp=lines.split(',')
#     temp1=temp[1].split(':')
#     temp2 = temp[2].split(':')
#     temp3 = temp[3].split(':')
#     name=temp[0][10:-1]
#     time=float(temp1[1][1:-1])
#     res=float(temp2[1][10:-4])
#     res_th=float(temp3[1][10:-4])
#     log_d[name]={}
#     log_d[name]['time']=time
#     log_d[name]['res'] = abs(int(res))
#     log_d[name]['res_th'] = abs(int(res_th))
#
# n=2000
# for p in np.linspace(0.001,0.01,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p
#
#
#
#
#
# y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])
#
#
# x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])
#
# nd=len(y_axis)
#
#
# plotm=np.zeros([2,nd])
#
# plotm[0,:]=x_axis
# plotm[1,:]=y_axis
#
# plotms=plotm[:, plotm[0].argsort()]
#
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=2000')
# plt.ylabel('Performance of HypOp for Maxind on random graphs')
# plt.xlabel('ln(p)')
# plt.legend(loc='upper right')
#
#
#
# # with open('maxind_rand_3000.log') as f:
# #     Log=f.readlines()
#
# with open('maxind_rand_3000.log') as f:
#     Log=f.readlines()
#
# log_d={}
# for lines in Log:
#     temp=lines.split(',')
#     temp1=temp[1].split(':')
#     temp2 = temp[2].split(':')
#     temp3 = temp[3].split(':')
#     name=temp[0][10:-1]
#     time=float(temp1[1][1:-1])
#     res=float(temp2[1][10:-4])
#     res_th=float(temp3[1][10:-4])
#     log_d[name]={}
#     log_d[name]['time']=time
#     log_d[name]['res'] = abs(int(res))
#     log_d[name]['res_th'] = abs(int(res_th))
#
# n=3000
# for p in np.linspace(0.001,0.01,10):
#     name = 'Random_' + str(int(1000 * p)) + '_Over1000_' + str(n) + '.txt'
#     log_d[name]['p']=p
#
#
#
#
# y_axis=np.array([log_d[name]['res_th']/log_d[name]['res'] for name in log_d])
#
#
# x_axis=np.array([np.log(log_d[name]['p']) for name in log_d])
#
# nd=len(y_axis)
#
#
# plotm=np.zeros([2,nd])
#
# plotm[0,:]=x_axis
# plotm[1,:]=y_axis
#
# plotms=plotm[:, plotm[0].argsort()]
#
# plt.plot(plotms[0,:],plotms[1,:], marker='o', label='N=3000')
# plt.ylabel('Performance of HypOp for Maxind on random graphs')
# plt.xlabel('ln(p)')
# plt.legend(loc='upper right')
#
#
# path='/Users/nasimeh/Documents/distributed_GCN-main-6/res/plots/'
# os.chdir(path)
#
# plt.savefig("phase_transit_random_lr_1e-4.png")
# plt.show()
#
#
#
# #lr=1e-4
# n_vec=[200, 1000, 2000, 3000]
# p_phase=[0.03, 0.009, 0.004, 0.003]
#
# #lr=1e-5
# n_vec=[200, 1000, 2000, 3000]
# p_phase=[0.03, 0.008, 0.004, 0.003]
#
# p_erdos=[np.log(item)/item for item in n_vec]
#
# plt.plot(n_vec,p_phase, marker='o', label='Phase Transition Curve')
# plt.ylabel('Transition Probability')
# plt.xlabel('n')
# plt.legend(loc='upper right')
#
# plt.plot(n_vec,p_erdos, marker='o', label='ln(N)/N Curve')
# plt.ylabel('Transition Probability')
# plt.xlabel('Number of Nodes')
# plt.legend(loc='upper right')
# plt.savefig("phase_transit_erdos_lr_1e-5.png")
# plt.show()