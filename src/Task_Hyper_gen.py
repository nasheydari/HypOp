import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA

from numpy.random import rand
import random
import csv
import pickle
from src.data_reading import read_hypergraph

#name="LCC_PRE_1993_to_1995.txt"
# name="LCC_PRE_2017_to_2019.txt"
# path="/Users/nasimeh/Documents/distributed_GCN-main-6/data/hypergraph_data/APS/3years_Stanford/"+name

name="Hyp_2664_3901.txt"
path="/Users/nasimeh/Documents/distributed_GCN-main-6/data/hypergraph_data/synthetic/new/"+name

constraints, header=read_hypergraph(path)


n=header['num_nodes']
m=header['num_constraints']


info = {x+1:[] for x in range(n)}
for constraint in constraints:
    for node in constraint:
        info[abs(node)].append(constraint)


ns_aug={x+1:[] for x in range(n)}
k=0
conss_aug={str(cons):{} for cons in constraints}

for i in range(1,n+1):
    for j in range(len(info[i])):
        k+=1
        ns_aug[i].append(k)
        cons=info[i][j]
        conss_aug[str(cons)][i]=k


n_aug=k

constrainst_aug=[]
for cons in constraints:
    cons_aug=list(conss_aug[str(cons)].values())
    cons_aug.append('T')
    constrainst_aug.append(cons_aug)

for i in range(1,n+1):
    cons=list(ns_aug[i])
    cons.append('A')
    constrainst_aug.append(cons)
header_aug=[n_aug, m]

#pth='/Users/nasimeh/Documents/distributed_GCN-main-6/data/task_data/APS/3years_Stanford/'+name

pth="/Users/nasimeh/Documents/distributed_GCN-main-6/data/task_data/synthetic/"+name


f=open(pth, 'w')
f.write(str(header_aug[0]))
f.write(' ')
f.write(str(header_aug[1]))
f.write('\n')


i=0

for cons in constrainst_aug:
    for ns in cons:
        i+=1
        f.write(str(ns))
        if i< len(cons):
            f.write(' ')
    f.write('\n')
    i=0





