import json

import networkx as nx
import yaml
from model import ElecNoStep


def str2int(json_data):
    
    new_dict = {}
    for key, value in json_data.items():
        new_dict[int(key)] = value
    return new_dict

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

def influenced_tl_by_elec(elec_state, elec2road, tgraph):
    elec10kv = []
    for key in ['ruined', 'cascaded', 'stopped']:
        elec10kv += elec_state[5][key]
    elec10kv = [str(node) for node in elec10kv if str(node) in elec2road.keys()]
    tl_id = [elec2road[node] for node in elec10kv if elec2road[node] in tgraph.nodes()]
    return tl_id

def calculate_size_of_gcc(Graph):

    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)]
    size_of_gcc = max(size_of_connected_components)

    return size_of_gcc

def calculate_pairwise_connectivity(Graph):

    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)] 
    element_of_pc  = [size*(size - 1)/2 for size in size_of_connected_components] 
    pairwise_connectivity = sum(element_of_pc)

    return pairwise_connectivity

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

def nodes_ranked_by_CI(Graph):
    """
    计算网络中各点的组合影响(CI),由大到小将点排序
    return->list
    """
    node_degree_map = dict(nx.degree(Graph))
    node_ci = []
    for node in node_degree_map:
        ci = 0
        neighbors = list(Graph.neighbors(node))
        for neighbor in neighbors:
            ci += (node_degree_map[neighbor]-1)
        node_ci.append((node,ci*(node_degree_map[node]-1)))
    list.sort(node_ci,key = lambda y:y[1],reverse=True)
    nodes = [example[0] for example in node_ci]

    return nodes
    
def nodes_ranked_by_Degree(Graph):
    """
    计算网络中各点的度(CI),由大到小将点排序
    return->list
    """
    node_degree = list(nx.degree(Graph))
    list.sort(node_degree,key = lambda y:y[1],reverse=True)
    nodes = [example[0] for example in node_degree]

    return nodes
