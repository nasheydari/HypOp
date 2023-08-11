# HypOp
Hypergraph Neural Network-Based Combinatorial Optimization

Create a jason config file in the config folder. Sample config file: ./config/Maxcut_R.json:

  log file will be saved in the "logging_path" address
  Optimization related data including graphs/hypergraphs have to be put in the "folder_path" address specified in the config file.

  "data":stanford in the jason file means the graph is represented as a text file with a header=(number of nodes, number of edges) and different lines for each edge

  

In order to run, run the file run.py: make sure to put the name of the config file in "with open('configs/maxcut_R.json') as f:".
