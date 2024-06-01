import networkx as nx
import matplotlib.pyplot as plt
import torch
import random

#MAX degree methodology

def coarsen1(constraints, header):
    G = nx.Graph()
    G.add_edges_from(constraints)

    nx.draw(G, with_labels=True)            
    plt.savefig('graph.png')
    num_node_org = len(G)
    index_node = 0

    for u, v, data in G.edges(data=True):
        data['weight'] = 1

    while (len(G)>=num_node_org/2):
        index_node += 1
        candidates1 = [u for u in G.nodes() if not isinstance(u, tuple)]
        if not candidates1:
            break

        u = max(candidates1, key=lambda e: G.degree(e))

        candidates2 = [v for v in G.neighbors(u) if not isinstance(v, tuple)]
        if candidates2: ##change
            v = min(candidates2, key=lambda e: G.degree(e))

            common_neighbors = set(G.neighbors(u)) & set(G.neighbors(v))
    
            for neighbor in common_neighbors:
                if G.has_edge(u, neighbor) and G.has_edge(v, neighbor):
                    G[u][neighbor]['weight'] = G[u][neighbor]['weight'] + G[v][neighbor]['weight']

            G = nx.contracted_nodes(G, u, v, self_loops=False)
            mapping = {u: (u,v)}
            G = nx.relabel_nodes(G, mapping)

        else:
            mapping = {u: (u,)}
            G = nx.relabel_nodes(G, mapping)

    
    nx.draw(G, with_labels=True)            
    plt.savefig('graph_res.png')
    #new_edges = [[u, v] for u, v in G.edges()]

    new_header = {}
    new_header['num_nodes'] = len(G)
    new_header['num_constraints'] = G.number_of_edges()

    G_copy = G.copy()

    graph_dict = {}
    index = 1
    for node in G_copy.nodes():
        mapping = {node: index}
        graph_dict[index] = node
        G_copy = nx.relabel_nodes(G_copy, mapping)
        index+=1
    new_edges = []
    new_weights = []
    for u, v, data in G_copy.edges(data=True):
        new_edges.append([u,v])
        new_weights.append(data['weight'])
    
    return new_header, new_edges, new_weights, graph_dict



# MIN degree methodology

def coarsen2(constraints, header):
    G = nx.Graph()
    G.add_edges_from(constraints)

    num_node_org = len(G)
    index_node = 0

    for u, v, data in G.edges(data=True):
        data['weight'] = 1

    while (len(G)>=num_node_org*(2/3)):
        index_node += 1
        candidates1 = [u for u in G.nodes() if not isinstance(u, tuple)]
        if not candidates1:
            break

        u = min(candidates1, key=lambda e: G.degree(e))

        candidates2 = [v for v in G.neighbors(u) if not isinstance(v, tuple)]
        if candidates2: 
            v = min(candidates2, key=lambda e: G.degree(e))

            common_neighbors = set(G.neighbors(u)) & set(G.neighbors(v))
    
            for neighbor in common_neighbors:
                if G.has_edge(u, neighbor) and G.has_edge(v, neighbor):
                    G[u][neighbor]['weight'] = G[u][neighbor]['weight'] + G[v][neighbor]['weight']

            G = nx.contracted_nodes(G, u, v, self_loops=False)
            mapping = {u: (u,v)}
            G = nx.relabel_nodes(G, mapping)

        else:
            mapping = {u: (u,)}
            G = nx.relabel_nodes(G, mapping)

    
    nx.draw(G, with_labels=True)            
    plt.savefig('graph_res.png')
    #new_edges = [[u, v] for u, v in G.edges()]

    new_header = {}
    new_header['num_nodes'] = len(G)
    new_header['num_constraints'] = G.number_of_edges()

    G_copy = G.copy()

    graph_dict = {}
    index = 1
    for node in G_copy.nodes():
        mapping = {node: index}
        graph_dict[index] = node
        G_copy = nx.relabel_nodes(G_copy, mapping)
        index+=1
    new_edges = []
    new_weights = []
    for u, v, data in G_copy.edges(data=True):
        new_edges.append([u,v])
        new_weights.append(data['weight'])
    
    return new_header, new_edges, new_weights, graph_dict

def coarsen3(constraints, header):
    G = nx.Graph()
    G.add_edges_from(constraints)

    num_node_org = len(G)
    index_node = 0

    for u, v, data in G.edges(data=True):
        data['weight'] = 1

    while (len(G)>=num_node_org*(2/3)):
        index_node += 1
        candidates1 = [u for u in G.nodes() if not isinstance(u, tuple)]
        if not candidates1:
            break

        u = min(candidates1, key=lambda e: G.degree(e))

        candidates2 = [v for v in G.neighbors(u) if not isinstance(v, tuple)]
        if candidates2: 
            v = min(candidates2, key=lambda e: G.degree(e))

            common_neighbors = set(G.neighbors(u)) & set(G.neighbors(v))
    
            for neighbor in common_neighbors:
                if G.has_edge(u, neighbor) and G.has_edge(v, neighbor):
                    G[u][neighbor]['weight'] = G[u][neighbor]['weight'] + G[v][neighbor]['weight']

            G = nx.contracted_nodes(G, u, v, self_loops=False)
            mapping = {u: (u,v)}
            G = nx.relabel_nodes(G, mapping)

        else:
            mapping = {u: (u,)}
            G = nx.relabel_nodes(G, mapping)

    new_header = {}


    G_copy = G.copy()

    graph_dict = {}
    index = 1

    for node in G_copy.nodes():
        mapping = {node: index}
        graph_dict[index] = node
        G_copy = nx.relabel_nodes(G_copy, mapping)
        index+=1

    new_edges = []
    new_weights = []

    n_prunning = int(1/5 * len(G_copy.edges(data=True)))
    n_pruned = 0

    for u, v, data in G_copy.edges(data=True):
        prune = random.choices([True, False], weights = [0.3, 0.7])
        if prune[0] and n_pruned<n_prunning and data['weight']==1:
            n_pruned = n_pruned + 1
        else:
            new_edges.append([u,v])
            new_weights.append(data['weight'])
    
    new_header['num_nodes'] = len(G)
    new_header['num_constraints'] = len(new_edges)
            
    
    return new_header, new_edges, new_weights, graph_dict

def coarsen4(constraints, header):
    G = nx.Graph()
    G.add_edges_from(constraints)

    num_node_org = len(G)
    index_node = 0

    while (len(G)>=num_node_org*(2/3)):
        index_node += 1
        candidates1 = [u for u in G.nodes() if not isinstance(u, tuple)]
        if not candidates1:
            break

        u = min(candidates1, key=lambda e: G.degree(e))

        candidates2 = [v for v in G.neighbors(u) if not isinstance(v, tuple)]
        if candidates2: 
            v = min(candidates2, key=lambda e: G.degree(e))

            G = nx.contracted_nodes(G, u, v, self_loops=False)
            mapping = {u: (u,v)}
            G = nx.relabel_nodes(G, mapping)

        else:
            mapping = {u: (u,)}
            G = nx.relabel_nodes(G, mapping)


    new_header = {}
    new_header['num_nodes'] = len(G)
    new_header['num_constraints'] = G.number_of_edges()

    G_copy = G.copy()

    graph_dict = {}
    index = 1
    for node in G_copy.nodes():
        mapping = {node: index}
        G_copy = nx.relabel_nodes(G_copy, mapping)
        if isinstance(node, tuple) and len(node)==2:
            graph_dict[node[0]] = (index, 0)
            graph_dict[node[1]] = (index, 1)
        elif isinstance(node, tuple) and len(node)==1:
            graph_dict[node[0]] = (index, 0)
        else:
            graph_dict[node] = (index, 0)
        index+=1

    new_edges = [[u, v] for u, v in G_copy.edges()]
    
    return new_header, new_edges, graph_dict


def coarsen5(constraints, header):
    G = nx.Graph()
    G.add_nodes_from(range(1, header['num_nodes']))
    G.add_edges_from(constraints)

    num_node_org = header['num_nodes']
    index = num_node_org + 1
    graph_dict = {}

    while (len(G)> num_node_org*(1/2)):
       
        candidates1 = [u for u in G.nodes() if u <= num_node_org]
        if not candidates1:
            break

        u = min(candidates1, key=lambda e: G.degree(e))

        candidates2 = [v for v in G.neighbors(u) if v <= num_node_org]

        if candidates2: 
            v = min(candidates2, key=lambda e: G.degree(e))

            G = nx.contracted_nodes(G, u, v, self_loops=False)
            mapping = {u: index}
            G = nx.relabel_nodes(G, mapping)
            graph_dict[u] = (index - num_node_org, 0)
            graph_dict[v] = (index - num_node_org, 1)

        else:
            mapping = {u: index}
            G = nx.relabel_nodes(G, mapping)
            graph_dict[u] = (index - num_node_org, 0)
        
        index += 1


    new_header = {}
    new_header['num_nodes'] = len(G)
    new_header['num_constraints'] = G.number_of_edges()

    for node in G.nodes():
        if node <= num_node_org:
            mapping = {node: index}
            G = nx.relabel_nodes(G, mapping)
            graph_dict[node] = (index - num_node_org, 0)
            index += 1


    new_edges = [[u - num_node_org, v - num_node_org] for u, v in G.edges()]
    
    return new_header, new_edges, graph_dict

def compute_prob_org(graph_dict, probs, n_org):
    x = probs.squeeze()
    prob_org = torch.tensor([0.0 for _ in range(n_org)])

    for node in graph_dict:
        if len(graph_dict[node]) == 2:
            prob_org[graph_dict[node][0] - 1] = x[node-1][0]
            prob_org[graph_dict[node][1] - 1] = x[node-1][1]
        else:
            prob_org[graph_dict[node][0] - 1] = x[node-1][0]

    return prob_org
