import matplotlib.pyplot as plt
import json
import os
import numpy as np
import re


path='/Users/nasimeh/Documents/distributed_GCN-main-6/log/'
os.chdir(path)







with open('maxind_powerlaw2.log') as f:
    Log=f.readlines()

log_power={}
log_reg={}
log_rand={}
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
    if re.search('random', name):
        power = float(name[18:-4].split('_')[0])
        log_rand[power] = {}
        log_rand[power]['p']= float(name[18:-4].split('_')[1])
        log_rand[power]['n']= 200
        log_rand[power]['time'] = time
        log_rand[power]['res'] = abs(int(res))
        log_rand[power]['res_th'] = abs(int(res_th))
    elif re.search('reg', name):
        power=float(name[15:-4].split('_')[0])
        log_reg[power]={}
        log_reg[power]['d']=int(name[15:-4].split('_')[1])
        log_reg[power]['n'] = 200
        log_reg[power]['time'] = time
        log_reg[power]['res'] = abs(int(res))
        log_reg[power]['res_th'] = abs(int(res_th))
    else:
        power=float(name[12:-4])
        log_power[power]={}
        log_power[power]['n'] = 200
        log_power[power]['time']=time
        log_power[power]['res'] = abs(int(res))
        log_power[power]['res_th'] = abs(int(res_th))





powers=sorted(log_power.keys())

y_axis1=np.array([log_power[name]['res_th']/log_power[name]['res'] for name in powers])
y_axis2=np.array([log_reg[name]['res_th']/log_reg[name]['res'] for name in powers])
y_axis3=np.array([log_rand[name]['res_th']/log_rand[name]['res'] for name in powers])



x_axis=powers
x_axis2=[log_rand[name]['p'] for name in powers]



nd=len(y_axis1)


plotm=np.zeros([4,nd])

plotm[0,:]=x_axis2
plotm[1,:]=y_axis1
plotm[2,:]=y_axis2
plotm[3,:]=y_axis3

plotms=plotm[:, plotm[0].argsort()]

plt.plot(plotms[0,:],plotms[1,:], marker='o', label='Powerlaw')
plt.plot(plotms[0,:],plotms[2,:], marker='x', label='Regular')
plt.plot(plotms[0,:],plotms[3,:], marker='+', label='Random')
plt.ylabel('GNN outcome over HypOp outcome')
plt.xlabel('P')
plt.legend(loc='upper right')
plt.savefig('/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/plots/Powerlaw_p.pdf')
plt.show()



