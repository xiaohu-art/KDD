import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(coo_matrix):
    
    coords = np.vstack((coo_matrix.row, coo_matrix.col)).transpose()
    values = coo_matrix.data
    shape = coo_matrix.shape
    return coords, values, shape

def vgae_loss(recover, adj, mu, logvar, node_num, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(recover, adj, pos_weight)
    KL = -0.5 / node_num * torch.mean(torch.sum( 1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost + KL

def mask_test_edges(orig_adj):
    triu_adj = sp.triu(orig_adj)
    edges, _, _ = sparse_to_tuple(triu_adj)

    print(len(edges))
