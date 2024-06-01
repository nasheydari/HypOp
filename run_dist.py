from src.run_exp import exp, exp_traditional, exp_centralized, exp_gradient, exp_centralized_for, exp_centralized_for_multi
from src.solver import QUBO_solver
import json
import torch.multiprocessing as mp


if __name__ == '__main__':  
   test_mode = "dist"
   dataset = "stanford"

   if test_mode == "infer":
      if dataset == "stanford":
         with open('infer_configs/maxcut_R_for.json') as f:
            params = json.load(f)
      elif dataset == "arxiv":
         with open('infer_configs/maxcut_arxiv_for.json') as f:
            params = json.load(f)
      exp_centralized(params)

   elif test_mode == "dist":
      if dataset == "stanford":
         with open('dist_configs/maxcut_R_for.json') as f:
            params = json.load(f)
      elif dataset == "arxiv":
         with open('dist_configs/maxcut_arxiv_for.json') as f:
            params = json.load(f)
      
      params["logging_path"] = params["logging_path"].split(".log")[0] +str(params["multi_gpu"]) + "_" + params["data"] + "_test.log"
      if params["multi_gpu"]:
         mp.spawn(exp_centralized_for_multi, args=(list(range(params["num_gpus"])), params), nprocs=params["num_gpus"])
      else:
         exp_centralized_for(params)