def read_stanford(path):
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    header = {}
    info = lines[0].split(' ')
    header['num_nodes'] = int(info[0])
    header['num_constraints'] = int(info[1])
    constraints = []
    for con in lines[1:-1]:
        temp = con.split(' ')
        constraints.append([int(x) for x in temp[:2]])
    return constraints, header


def read_hypergraph(path):
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    header = {}
    info = lines[0].split(' ')
    header['num_nodes'] = int(info[0])
    header['num_constraints'] = int(info[1])
    #weights=[]
    constraints = []
    i=0
    for con in lines[1:-1]:
        temp = con.split(' ')
        #weights[i]=temp[0]
        constraints.append([int(x) for x in temp])
        i += 1
    #return constraints, weights, header
    return constraints, header

def read_hypergraph_task(path):
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    header = {}
    info = lines[0].split(' ')
    header['num_nodes'] = int(info[0])
    header['num_constraints'] = int(info[1])
    #weights=[]
    constraints = []
    i=0
    for con in lines[1:-1]:
        temp = con.split(' ')
        #weights[i]=temp[0]
        cons=[int(x) for x in temp[:-1]]
        cons.append(temp[-1])
        constraints.append(cons)
        i += 1
    #return constraints, weights, header
    return constraints, header

def read_uf(path):
    with open(path) as f:
        file = f.read()
    lines = file.split('\n')
    header = {}
    for i in range(len(lines)):
        if lines[i][0] == 'p':
            info = lines[i].split(' ')
            info = [x for x in info if len(x) > 0]
            header['type'] = info[1]
            header['num_nodes'] = int(info[2])
            header['num_constraints'] = int(info[3])
            materil = ' '.join(lines[i+1:(i+1+header['num_constraints'])]) + ' '
            break
    constraints_words = materil[1:].split(' 0 ')
    constraints = []
    for con in constraints_words:
        temp = con.split(' ')
        if all([x.lstrip('-').isnumeric() for x in temp]):
            constraints.append([int(x) for x in temp])
    return constraints, header


def read_NDC(path):
    path1=path+'/NDC-substances-full-nverts.txt'
    path2=path+'/NDC-substances-full-simplices.txt'

    with open(path1) as f:
        file = f.read()
    lines = file.split('\n')[:-1]
    with open(path2) as f2:
        file2 = f2.read()
    lines2 = file2.split('\n')[:-1]

    constraints=[]
    i=0
    for j in lines:
        hyperedge=[int(lines2[l]) for l in range(i,i+int(j))]
        i+=int(j)
        constraints.append(hyperedge)
    constraints2 = [cons for cons in constraints if len(cons) > 1]
    n = len(set(lines2))
    info = {x + 1: [] for x in range(n)}
    for constraint in constraints2:
        for node in constraint:
            info[abs(node)].append(constraint)
    nodes_n=[i for i in range(1,n+1) if len(info[i]) > 0]
    ni=1
    nodec_n_dic={}
    for j in nodes_n:
        nodec_n_dic[j]=ni
        ni+=1
    constraints2_n=[]
    for cons in constraints2:
        cons_n=[nodec_n_dic[j] for j in cons]
        constraints2_n.append(cons_n)
    m=len(constraints2_n)
    n=len(nodes_n)
    header = {}
    header['num_nodes']=n
    header['num_constraints'] = m

    return constraints2_n, header

def read_arxiv():
    # load ogbn-arxiv in ogb & dgl format
    from ogb.nodeproppred import DglNodePropPredDataset
    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    
    header = {}
    header['num_nodes'] = dataset.graph[0].number_of_nodes()
    header['num_constraints'] = dataset.graph[0].number_of_edges()

    edge_connections = dataset.graph[0].edges()
    edge_connections = [edge_connections[0]+1, edge_connections[1]+1]
    edge_connections = [edge_connections[0].tolist(), edge_connections[1].tolist()]
    # convert edge_connections to list of lists
    constraints = [[edge_connections[0][x], edge_connections[1][x]] for x in range(len(edge_connections[0]))]
    return constraints, header