import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path):
        self.graph = None
        self.feat = None
        self.node_list = None

    @property
    def node_num(self):
        return self.graph.num_nodes()

    @property
    def egde_num(self):
        return self.graph.num_edges()

    def build_graph(self):
        pass

    def build_feat(self, embed_dim, hid_dim, feat_dim, 
                    k, epochs,
                    pt_path):

        print('trainging features ...')
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
                        " time=", "{:.4f}s".format(time.time() - t)
                        )

        feat = gcn.sage(self.graph, self.graph.ndata['feat'])
        try:
            torch.save(feat, pt_path)
            print("saving features sucess")
        except:
            print("saving features failed")
        return feat

class ElecGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path):
        super().__init__(file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path)
        print('electricity network construction!')
        self.node_list, self.graph = self.build_graph(file)
        try:
            feat = torch.load(pt_path)
            print('features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim, 
                                         khop, epochs,
                                         pt_path)

    def build_graph(self, file):

        print('building elec graph ...')
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

        node_list : dict = {i:j for i,j in enumerate(list(G.nodes()))}
        print('graph builded.')
        return node_list, dgl.from_networkx(G)

class TraGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path):
        super().__init__(file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path)
        print('traffice network constrcution!')
        self.graph = self.build_graph(file)
        try:
            feat = torch.load(pt_path)
            print('features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim, 
                                         khop, epochs,
                                         pt_path)

    def build_graph(self, file):

        print('building traffic graph ...')
        with open(file, 'r') as f:
            data = json.load(f)
        G = nx.Graph()
        for road, junc in data.items():
            if len(junc) == 2:
                G.add_edge(junc[0], junc[1], id=int(road))

        node_list : dict = {i:j for i,j in enumerate(list(G.nodes()))}
        return node_list, dgl.from_networkx(G)

