

import time
import argparse


import numpy as np
import torch
from colorama import Back, Fore, Style, init
from model import DQN, Bigraph, ElecGraph, TraGraph
from utils import (calculate_pairwise_connectivity, influenced_tl_by_elec,
                   init_env,nodes_ranked_by_Degree,nodes_ranked_by_CI)

FILE = './data/e10kv2tl.json'
EFILE = './data/electricity/all_dict_correct.json'
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
ept = './embedding/elec_feat.pt'
tpt = './embedding/tra_feat.pt'
bpt = ('./embedding/bifeatures/bi_elec_feat.pt', './embedding/bifeatures/bi_tra_feat.pt') 
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5

if __name__ == "__main__":
    egraph = ElecGraph(file=EFILE,
                embed_dim=EMBED_DIM,
                hid_dim=HID_DIM,
                feat_dim=FEAT_DIM,
                khop=KHOP,
                epochs=500,
                pt_path=ept)

    tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)
    bigraph = Bigraph(efile=EFILE, tfile1=TFILE1, tfile2=TFILE2, tfile3=TFILE3, file=FILE,
                embed_dim=EMBED_DIM,
                hid_dim=HID_DIM,
                feat_dim=FEAT_DIM,
                subgraph = (egraph, tgraph),
                khop=KHOP,
                epochs=600,
                pt_path=bpt)

    elec_nodes_degree = np.loadtxt('/data5/maojinzhu/KDD_bigragh/data/electricity/nodes_ranked_by_Degree.txt')
    elec_nodes_degree = [int(node) for node in elec_nodes_degree]
    elec_nodes_CI = np.loadtxt('/data5/maojinzhu/KDD_bigragh/data/electricity/nodes_ranked_by_CI.txt')
    elec_nodes_CI = [int(node) for node in elec_nodes_CI ]
    road_nodes_degree = np.loadtxt('/data5/maojinzhu/KDD_bigragh/data/road/nodes_ranked_by_degree.txt')
    road_nodes_degree = [int(node) for node in road_nodes_degree]
    road_nodes_CI = np.loadtxt('/data5/maojinzhu/KDD_bigragh/data/road/nodes_ranked_by_CI.txt')
    road_nodes_CI = [int(node) for node in road_nodes_CI ]
    num_elec = 25
    num_road = 25

    tgc = tgraph.nxgraph.copy()
    reward = []
    elec_env = init_env()
    original_power = elec_env.ruin([])
    origin_val = calculate_pairwise_connectivity(tgc)
    t_val = 1
    tpower = original_power
    total_reward = 0

    for i in range(num_road):
        h_val = t_val
        tgc.remove_node(road_nodes_degree[i])
        t_val = calculate_pairwise_connectivity(tgc) / origin_val
        total_reward += (0.5*(h_val - t_val)*1e4)
        reward.append(total_reward)
    for i in range(num_elec):
        h_val = t_val
        hpower = tpower
        tpower,elec_state = elec_env.ruin([elec_nodes_degree[i]],flag=0)
        nodes = influenced_tl_by_elec(elec_state, bigraph.elec2road, tgraph.nxgraph)
        tgc.remove_nodes_from(nodes)
        t_val = calculate_pairwise_connectivity(tgc) / origin_val
        total_reward += (0.5*(hpower - tpower)/1e5)
        total_reward += (0.5*(h_val - t_val)*1e4)
        reward.append(total_reward)
    np.savetxt('./result/degree_25_25.txt',np.array(reward))

    tgc = tgraph.nxgraph.copy()
    reward = []
    elec_env = init_env()
    original_power = elec_env.ruin([])
    origin_val = calculate_pairwise_connectivity(tgc)
    t_val = 1
    tpower = original_power
    total_reward = 0

    for i in range(num_road):
        h_val = t_val
        tgc.remove_node(road_nodes_CI[i])
        t_val = calculate_pairwise_connectivity(tgc) / origin_val
        total_reward += (0.5*(h_val - t_val)*1e4)
        reward.append(total_reward)
    for i in range(num_elec):
        h_val = t_val
        hpower = tpower
        tpower,elec_state = elec_env.ruin([elec_nodes_CI[i]],flag=0)
        nodes = influenced_tl_by_elec(elec_state, bigraph.elec2road, tgraph.nxgraph)
        tgc.remove_nodes_from(nodes)
        t_val = calculate_pairwise_connectivity(tgc) / origin_val
        total_reward += (0.5*(hpower - tpower)/1e5)
        total_reward += (0.5*(h_val - t_val)*1e4)
        reward.append(total_reward)
    np.savetxt('./result/CI_25_25.txt',np.array(reward))





    
    

