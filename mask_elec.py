from model import ElecGraph, GCN, construct_negative_graph, compute_loss
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
# parser.add_argument('--feat', type=str, required=True, help='pre-train feat or random feat')
# parser.add_argument('--label', type=str, required=True, help='train or test')

args = parser.parse_args()

ept = './embedding/elec_feat.pt'
perturb_pth = './embedding/perturb_0.pt'

EFILE = './data/electricity/all_dict_0.json'
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

    orig_feat = torch.load(ept).detach()

    egraph = ElecGraph(file=EFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

    node_num = egraph.node_num

    perturb_graph = egraph.graph
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
        loss = compute_loss(pos_score, neg_score) + 0.1 * dist
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

    

    