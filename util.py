import networkx as nx
import numpy as np
import torch

from option import Options


def get_options():
    opt = Options().initialize()
    return opt


def topological_measures_pos_or_neg(data):
    # ROI is the number of brain regions (i.e.,200 in our case)
    opt = get_options()
    p_CC = np.empty((0, opt.in_dim), int)
    n_CC = np.empty((0, opt.in_dim), int)
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            if j == 0:
                A = data[i][j].cpu().detach().numpy()
                A = np.where(A < opt.adj_thresh, 0., A)
                np.fill_diagonal(A, 0)
                G = nx.from_numpy_matrix(A)
                U = G.to_undirected()
                ec = nx.eigenvector_centrality_numpy(U)
                # bc = nx.betweenness_centrality(U)
                closeness_centrality = np.array([ec[g] for g in U])
                n_CC = np.vstack((n_CC, closeness_centrality))
                # print('i: ',i,' n_EC: ',n_CC)
                # n_CC.append(closeness_centrality)
            else:
                A = data[i][j].cpu().detach().numpy()
                A = np.where(A < opt.adj_thresh, 0., A)
                np.fill_diagonal(A, 0)
                G = nx.from_numpy_matrix(A)
                U = G.to_undirected()
                ec = nx.eigenvector_centrality_numpy(U)
                # bc = nx.betweenness_centrality(U)
                closeness_centrality = np.array([ec[g] for g in U])
                p_CC = np.vstack((p_CC, closeness_centrality))

    n_EC = torch.tensor(n_CC).cuda()
    p_EC = torch.tensor(p_CC).cuda()

    return n_EC, p_EC

def binary(adj_):
    opt = get_options()
    bdj = adj_.detach()
    d = torch.zeros_like(bdj)
    b = torch.ones_like(bdj)
    cdj = torch.where(torch.abs(bdj) >= opt.adj_thresh, b, d)
    return cdj