from model import TraGraph, GCN, construct_negative_graph, compute_loss
from utils import init_env

import time
import torch
import torch.nn as nn
import dgl
import os
import random
import numpy as np
import networkx as nx
import argparse

parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.01, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')

args = parser.parse_args()

tpt = './embedding/primary.pt'
perturb_pth = './embedding/perturb_primary.pt'

TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch
GAMMA = args.gamma
EPSILON = args.epsilon
MEMORY_CAPACITY = 1000
TARGET_REPLACE_ITER = 25

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='primary',
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)

    edge_num = int(tgraph.egde_num * 0.9)

    node_num = tgraph.node_num

    tnodes = list(tgraph.nxgraph.nodes())
    tedges = list(tgraph.nxgraph.edges())
    # sample_nodes = random.sample(tnodes, 20)
    # sample_src = sample_nodes[:10]
    # sample_dst = sample_nodes[-10:]
    # sample_edge = [e for e in zip(sample_src, sample_dst)]

    # tedges = tedges + sample_edge
    random.shuffle(tedges)
    tedges = tedges[:edge_num]

    perturb_graph = nx.Graph()
    perturb_graph.add_nodes_from(tnodes)
    perturb_graph.add_edges_from(tedges)
    perturb_graph = dgl.from_networkx(perturb_graph)
    
    orig_feat = tgraph.feat.detach()
    embedding = nn.Embedding(node_num, EMBED_DIM, max_norm=1)
    perturb_graph.ndata['feat'] = embedding.weight

    gcn = GCN(EMBED_DIM, HID_DIM, FEAT_DIM)
    optimizer = torch.optim.Adam(gcn.parameters())
    optimizer.zero_grad()

    for epoch in range(2000):
        t = time.time()
        negative_graph = construct_negative_graph(perturb_graph, 5)
        pos_score, neg_score = gcn(perturb_graph, negative_graph, perturb_graph.ndata['feat'])
        feat = gcn.sage(perturb_graph, perturb_graph.ndata['feat'])


        dist = torch.dist(feat, orig_feat, p=2)
        
        
        loss = compute_loss(pos_score, neg_score) + dist
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                        " time=", "{:.4f}s".format(time.time() - t)
                        )
        
    print(" train_loss = ", "{:.5f} ".format(loss.item()))

    feat = gcn.sage(perturb_graph, perturb_graph.ndata['feat'])
    torch.save(feat, perturb_pth)

    