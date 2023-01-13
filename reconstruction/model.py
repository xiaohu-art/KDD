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

class Graph():
    def __init__(self, file):
        pass

    def build_graph(self):
        pass

    def build_feat(self):
        pass

class ElecGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim, khop, epochs):
        super().__init__(file)
        print('electricity networl construction !')
        self.graph = self.build_graph(file)
        self.feat = self.build_feat(embed_dim, hid_dim, feat_dim, 
                                    khop, epochs)

    @property
    def node_num(self):
        return self.graph.num_nodes()

    @property
    def egde_num(self):
        return self.graph.num_edges()

    def build_graph(self, file):

        print('building graph ...')
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

        print('graph builded.')
        return dgl.from_networkx(G)

    def build_feat(self, embed_dim, hid_dim, feat_dim, 
                    k, epochs):
        try:
            feat = torch.load('../embedding/elec_feat.pt')
            print('features loaded')
            return feat
        except:
            print('trainging elec features ...')
            embedding = nn.Embedding(self.node_num, embed_dim, max_norm=1)
            self.graph.ndata['feat'] = embedding.weight

            gcn = GCN(embed_dim, hid_dim, feat_dim)
            optimizer = torch.optim.Adam(gcn.parameters())
            optimizer.zero_grad()

            for epoch in range(epochs):
                t = time.time()
                negative_graph = construct_negative_graph(self.graph, k)
                pos_score, neg_score = gcn(self.graph, negative_graph, self.graph.ndata['feat'])
                loss = compute_loss(pos_score, neg_score)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                            " time=", "{:.5f}s".format(time.time() - t)
                            )

            feat = gcn.sage(self.graph, self.graph.ndata['feat'])
            try:
                torch.save(feat, '../embedding/elec_feat.pt')
                print("saving features sucess")
            except:
                print("saving features failed")
            return feat

FILE = '../data/elec_flow_input.json'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = 500

if __name__ == "__main__":

    elec = ElecGraph(file=FILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=EPOCH)
    