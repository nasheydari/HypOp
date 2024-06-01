from src.data_reading import read_uf, read_stanford, read_hypergraph, read_hypergraph_task
from src.solver import local_solver, centralized_solver, centralized_solver_for
import logging
import os
import h5py
import numpy as np


file_name="/Users/nasimeh/Documents/distributed_GCN-main-6/res/task_syn.hdf5"
f1 = h5py.File(file_name,'r+')

print(f1.keys)