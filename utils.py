import yaml
import json
import random
import os
import networkx as nx

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

def calculate_size_of_gcc(Graph):

    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)]
    size_of_gcc = max(size_of_connected_components)

    return size_of_gcc

def calculate_pairwise_connectivity(Graph):

    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)] 
    element_of_pc  = [size*(size - 1)/2 for size in size_of_connected_components] 
    pairwise_connectivity = sum(element_of_pc)

    return pairwise_connectivity

def influenced_tl_by_elec(elec_state, elec2road, tgraph):
    elec10kv = []
    for key in ['ruined', 'cascaded', 'stopped']:
        elec10kv += elec_state[5][key]
    elec10kv = [str(node) for node in elec10kv if str(node) in elec2road.keys()]
    tl_id = [elec2road[node] for node in elec10kv if elec2road[node] in tgraph.nodes()]
    return tl_id

def mask(graph, mask_dir):

    edges = list(graph.edges())
    edges = [(u, v) for (u, v) in edges if u > 2e8 and v > 2e8]
    random.shuffle(edges)

    len_9 = int(graph.number_of_edges() * 0.9)

    len_7 = int(graph.number_of_edges() * 0.7)

    edges_9 = edges[:len_9]
    edges_7 = edges[:len_7]

    g_9 = nx.Graph()
    g_9.add_nodes_from(graph.nodes())
    g_9.add_edges_from(edges_9)
    
    g_7 = nx.Graph()
    g_7.add_nodes_from(graph.nodes())
    g_7.add_edges_from(edges_7)

    g9_pth = os.path.join(mask_dir, 'g9.gpickle')
    g7_pth = os.path.join(mask_dir, 'g7.gpickle')

    nx.write_gpickle(g_9, g9_pth)
    nx.write_gpickle(g_7, g7_pth)
    