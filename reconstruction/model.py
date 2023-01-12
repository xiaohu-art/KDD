import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import networkx as nx
import numpy as np
import json
import time
import dgl
import dgl.nn as dnn
import dgl.function as dfn

class Graph():
    def __init__(self):
        self.graph = None
        self.adj = None
        self.feat = None

    def build_graph(self):
        pass

    def build_adj(self):
        pass

    def build_feat(self):
        pass

class ElecGraph(Graph):
    def __init__(self, file, embed_dim):
        super().__init__()
        self.graph = self.build_graph(file)
        self.node_num = self.graph.num_nodes()
        self.edge_num = self.graph.num_edges()
        self.build_feat(embed_dim)
        self.feat = None

    def build_graph(self, file):

        with open(file, 'r') as f:
            data = json.load(f)
        edges = []
        for element in data:
            prop = element.get('properties')
            if prop !=  None:
                edge = prop.get('relation')
                if edge !=  None:
                    edges.append(edge)

        G = nx.Graph()
        G.add_edges_from(edges)

        return dgl.from_networkx(G)

    def build_feat(self, embed_dim):
        embedding = nn.Embedding(self.node_num, embed_dim, max_norm=1)
        self.graph.ndata['feat'] = embedding.weight

        return 

class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = dnn.SAGEConv(in_dim, hid_dim, 'mean')
        self.conv2 = dnn.SAGEConv(hid_dim, out_dim, 'mean')
        self.relu = nn.ReLU()

    def forward(self, graph, input):
        output = self.conv1(graph, input)
        output = self.relu(output)
        output = self.conv2(graph, output)

        return output

class Innerproduct(nn.Module):
    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['feat'] = feat
            graph.apply_edges(dfn.u_dot_v('feat', 'feat', 'score'))
            return graph.edata['score']

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.sage = SAGE(in_dim, hid_dim, out_dim)
        self.pred = Innerproduct()

    def forward(self, graph, neg_graph, feat):
        feat = self.sage(graph, feat)
        return self.pred(graph, feat), self.pred(neg_graph, feat)

def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

def compute_loss(pos_score, neg_score):
        n_edges = pos_score.shape[0]

        return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


FILE = '../data/elec_flow_input.json'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
EPOCH = 500

if __name__ == "__main__":

    elec = ElecGraph(file=FILE,
                    embed_dim=EMBED_DIM)

    k = 5
    gcn = GCN(EMBED_DIM, HID_DIM, FEAT_DIM)
    optimizer = torch.optim.Adam(gcn.parameters())
    optimizer.zero_grad()

    for epoch in range(EPOCH):
        
        t = time.time()
        negative_graph = construct_negative_graph(elec.graph, k)
        pos_score, neg_score = gcn(elec.graph, negative_graph, elec.graph.ndata['feat'])
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                    " time=", "{:.5f}".format(time.time() - t)
                    )

    elec.feat = gcn.sage(elec.graph, elec.graph.ndata['feat'])
    try:
        torch.save(elec.feat, '../embedding/elec_feat.pt')
        print("saving features sucess")
    except:
        print("saving features failed")