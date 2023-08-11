from src.data_reading import read_uf, read_stanford, read_hypergraph
from src.solver import local_solver, centralized_solver
import logging
import os
import h5py
import numpy as np

def exp(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            else:
                log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')
            res, time = local_solver(constraints, header, params, log)
            print(np.average(res))
            print(min(res))
            if params['mode']=='maxind':
                N = 10000
                print((np.average(res)) / (N*0.45537))
            log.info(f'{file_name}:, running time: {time}, res: {res}')
            f.create_dataset(f"{file_name}", data = res)
#from src.traditional_solver import solve_maxsat_z3, solve_maxsat_FM, solve_uf_unknown
import timeit
def exp_traditional(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            num_vio, model = solve_maxsat_FM(path)
            time = timeit.default_timer() - temp_time
            log.info(f'{file_name}:, running time: {time}, res: {num_vio}')
            f.create_dataset(f"{file_name}", data = num_vio)

def exp_centralized(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg":
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                else:
                    log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

                res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver(constraints, header, params, file_name)

                #name= 'probs'+'_'+file_name+'.txt'
                #np.savetxt(name, probs)
                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))
                if params['mode']=='maxind':
                    N = 200
                    print((np.average(res)) / (N*0.45537))
                f.create_dataset(f"{file_name}", data = res)
from src.solver import gradient_solver  
def exp_gradient(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            else:
                log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')
            res, _ = gradient_solver(constraints, header, params)
            time = timeit.default_timer() - temp_time
            log.info(f'{file_name}:, running time: {time}, res: {res}')
            f.create_dataset(f"{file_name}", data = res)           



