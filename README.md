#HyoOp

Here you can find the code for HypOp, a tool for combinatorial optimization that employs hypergraph neural networks. It is versatile and can address a range of constrained optimization problems.
In the current version, we have included the following problems: graph and hypergraph MaxCut, graph MIS, SAT, and Resource Allocation (see paper for details). To add new problems, add the appropriate loss function in the loss.py file and add the appropriate function in data_reading.py to read your specific dataset. 

#### Install Required Packages

```bash
pip install -r dependency.txt
```



For single GPU training:

Set the config file in configs directory and run "run.py" with the specified config file.

## Parameters/Configs

#### Mode Parameters

    - data: uf/stanford/hypergraph/NDC
    - mode: maxcut/maxind/sat/task_vec/QUBO
   
    - Adam: true/false; false default
   

#### Training Parameters
    - lr: learning rate
    - epoch: number of training epochs
   - tol: training loss tolerace
   - patience: training patience
   - GD: false/true (true for direct optimization with gradient descent) 
   - load_G: false/true (true for when G is already computed and saved and want to load it)
    -  sparsify: false/true (true for when the graph is too dense and need to sparsify it)
    - sparsify_p: the probability of removing an edge if sparsify is true

#### Utils Parameters
    - mapping: threshold/distribution
      threshold: trivial mapping that maps numbers less than 0.5 to 0 and greater than 0.5 to 1
      distribution: mapping using simulated annealing
    
        
    - N_realize: only used when mapping = distribution: number of realizations from the distribution
    - Niter_h: only used when mapping = distribution: number of simulated annealing iterations
    - t: simulated annealing initial temperature
    
    - random_init: initializing simulated annealing randomly (not with HyperGNN)
    
    - logging_path:  path that the log file is saved
    - res_path: path that the result file is saved
    - folder_path: directory containing the data

    
   
#### Transfer learning

	- model_save_path: directory to save the model
    	- model_load_path: directory to load the model
	- transfer: false/true (true for transfer learning)
	- initial_transfer: false/true (true for initializing the models with a pre-trained model)
	

#### Sampling Parameters: for black-box ADANS optimization

    - K: 1 default: number of optimization rounds
    - num_samples: 1 default
    - minimum_good_samples: 4 default
    - random_portion: 0.6 default
    - local_portion: 0.2 default
    - cross_portion: 0.2 default
    
    
    
### For Multi-GPU training:

#### Run Distributed GPU Training

In `run_dist.py`, set dataset` variable to `stanford` for stanford dataset results, to `arxiv` for ogbn-arxiv dataset.

##### Step 1: Distributed Training

In `run_dist.py`, set `test_mode` variable to `dist`

```python
python -m torch.distributed.launch run.py
```

##### Step2: Postprocessing

in configs, set "load best out" to true, set "epoch" to 0

In `run_dist.py`, set `test_mode` variable to `infer`

```python
python -m torch.distributed.launch run.py
```

Note that the results generated in Step1 **is not** the final results, you have to run Step 2 for postprocessing. 
