import gurobipy as gp
from gurobipy import GRB
import os
import h5py
import timeit
import logging
from src.data_reading import read_uf, read_stanford, read_hypergraph, read_hypergraph_task, read_NDC
import json
import logging


with open('configs/Hypermaxcut_syn_new_single.json') as f:
   params = json.load(f)


folder_path = params['folder_path']
folder_length = len(os.listdir(folder_path))
logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
log = logging.getLogger('main')
print(f'Found {folder_length} files. Start experiments')
with h5py.File(params['res_path'], 'w') as f:
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg" or params['data'] == "bipartite" :
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            elif params['data'] == "task":
                constraints, header = read_hypergraph_task(path)
            elif params['data'] == "NDC":
                constraints, header = read_NDC(path)
            else:
                log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')




m = gp.Model("matrix1")


x = m.addMVar(shape=header['num_nodes'], vtype=GRB.BINARY, name="x")


#hypermaxcut objective

obj = gp.QuadExpr()

for c in constraints:
    temp_1s = 1
    temp_0s = 1

    for index in c:
        temp_1s = temp_1s*(1 - x[index - 1])
        temp_0s = temp_0s*(x[index - 1])


    obj = (temp_1s + temp_0s)



m.setObjective(obj, GRB.MAXIMIZE)


m.optimize()


print('Obj: %g' % m.ObjVal)

# for v in m.getVars():
#     print('%s %g' % (v.VarName, v.X))