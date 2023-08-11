from src.run_exp import exp, exp_traditional, exp_centralized, exp_gradient
from src.solver import QUBO_solver
import json

# with open('./configs/maxind_example_single_QUBO.json') as f:
#     params = json.load(f)
# exp(params)

# with open('./configs/maxcut_gradient_single.json') as f:
#    params = json.load(f)
# exp_gradient(params)

# with open('./configs/maxind_example_single_stanford.json') as f:
#    params = json.load(f)
# exp_centralized(params)

# with open('./configs/maxcut_example_s2_p1.json') as f:
#    params = json.load(f)
# exp_centralized(params)

# with open('./configs/maxind_example_s2_p1.json') as f:
#    params = json.load(f)
# exp_centralized(params)

#
# with open('./configs/maxind_example_p1_rand.json') as f:
#    params = json.load(f)
# exp_centralized(params)
#
# with open('./configs/maxcut_single_p1_reg.json') as f:
#    params = json.load(f)
# exp_centralized(params)


# with open('./configs/maxind_single_reg100.json') as f:
#    params = json.load(f)
#    exp_centralized(params)

#the Qubo solver for Maxind
# with open('./configs/maxind_p1_reg.json') as f:
#    params = json.load(f)
# QUBO_solver(params)


with open('configs/maxcut_R.json') as f:
   params = json.load(f)
exp_centralized(params)