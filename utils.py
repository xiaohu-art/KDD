import yaml
import json
import networkx as nx
import torch
import dgl

from model import ElecNoStep

def str2int(json_data):
    
    new_dict = {}
    for key, value in json_data.items():
        new_dict[int(key)] = value
    return new_dict

def numbers_to_etypes(num):
            switcher = {
                0: ('power', 'elec', 'power'),
                1: ('power', 'eleced-by', 'power'),
                2: ('junc', 'tran', 'junc'),
                3: ('junc', 'traned-by', 'junc'),
                4: ('junc', 'supp', 'power'),
                5: ('power', 'suppd-by', 'junc'),
            }

            return switcher.get(num, "wrong!")

def compute_loss(pos_score, neg_score):
        n_edges = pos_score.shape[0]

        return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def init_env():

    with open("./data/electricity/config.yml") as f:
            config = yaml.safe_load(f)
       
    with open('./data/electricity/power_10kv.json') as json_file:
            power_10 = json.load(json_file)

    with open('./data/electricity/power_110kv.json') as json_file:
        power_load = json.load(json_file)
        
    power_10 = str2int(power_10)
    power_load = str2int(power_load)
    
    with open('./data/electricity/all_dict_correct.json') as json_file:
        topology = json.load(json_file)
    
    for key in topology:
        topology[key] = str2int(topology[key])
    elec = ElecNoStep(config, topology, power_10, power_load)
    
    return elec

def calculate_pairwise_connectivity(removal_nodes,Graph):

    graph = Graph.copy()
    graph.remove_nodes_from(removal_nodes)
    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(graph)] # 计算各连通分量大小
    element_of_pc  = [size*(size - 1)/2 for size in size_of_connected_components] 
    pairwise_connectivity = sum(element_of_pc)

    return pairwise_connectivity

def calculate_size_of_gcc(removal_nodes,Graph):

    graph = Graph.copy()
    graph.remove_nodes_from(removal_nodes)
    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(graph)] # 计算各连通分量大小
    size_of_gcc = max(size_of_connected_components)

    return size_of_gcc

def calculate_anc(removal_nodes,Graph,connectivity = 'pc'):
    """
    accumulated normalized connectivity
    Parameters
        removal_node: a sequence of nodes(list)
        Graph: network(networkx)
        connectivity: predifined connectivity method(str->'pc','gcc')
    return anc(float->[0,1])
    """
    number_of_nodes = len(removal_nodes)
    if number_of_nodes == 0:
        return 1
    if connectivity == 'pc':
        connectivity_part = [calculate_pairwise_connectivity(removal_nodes[:i], Graph) for i in range(number_of_nodes)] 
        connectivity_all = calculate_pairwise_connectivity([],Graph)
    elif connectivity == 'gcc':
        connectivity_part = [calculate_size_of_gcc(removal_nodes[:i], Graph) for i in range(number_of_nodes)] 
        connectivity_all = calculate_size_of_gcc([],Graph)
    anc = sum(connectivity_part)/connectivity_all/number_of_nodes

    return anc
