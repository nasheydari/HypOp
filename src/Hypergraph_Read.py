import json
import numpy as np
# Import Module
import os
import copy
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA

from numpy.random import rand
import random
import csv
import pickle


n=500
m=1000

d=10

B=np.random.choice(d, n, replace=True)

I=np.zeros([n,m])
for i in range(n):
    edges=np.random.choice(m, B[i], replace=False)
    I[i,edges]=1

#I = np.loadtxt('Hypergraph_Inc.txt')

k = np.sum(I, axis=0)
I = np.copy(I[:,k>1]);



k = np.sum(I, axis=1)
I = np.copy(I[k>0,:]);


n,m=I.shape


np.savetxt('Hypergraph_Inc3.txt',I)

constraints=[]

for i in range(m):
    nodes = list(np.argwhere(I[:,i] > 0).T[0])
    constraints.append(nodes)


header=[n,m]

f=open('hypergraph_data/Hypergraph_constraints3.txt', 'w')
f.write(str(header[0]))
f.write(' ')
f.write(str(header[1]))
f.write('\n')

i=0

for cons in constraints:
    if len(cons)>1:
        for n in cons:
            i+=1
            f.write(str(n+1))
            if i< len(cons):
                f.write(' ')
        f.write('\n')
    i=0
