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