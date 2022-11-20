import sys
import numpy as np


num_node = 25
self_link = [(i, i) for i in range(num_node)]
# spatial
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
extward_ori_index = [(4, 16), (4, 20), (4, 22), (4, 24),
                     (16, 4), (16, 20), (16, 22), (16, 24),
                     (20, 4), (20, 16), (20, 22), (20, 24),
                     (22, 4), (22, 16), (22, 20), (22, 24),
                     (24, 4), (24, 16), (24, 20), (24, 20)]
extward = [(i - 1, j - 1) for (i, j) in extward_ori_index]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# extension
inward_ext_index = [(1, 2), (1, 3), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

# part
inward_part_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (6, 5), (7, 6),
                    (8, 7), (10, 9), (11, 10), (12, 11),
                    (14, 13), (15, 14), (16, 15), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_part = [(i - 1, j - 1) for (i, j) in inward_part_ori_index]
outward_part = [(j, i) for (i, j) in inward_part]
neighbor_part = inward_part + outward_part


edge = self_link + inward

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_ext_graph(num_node, self_link, inward, outward, extward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Ext = normalize_digraph(edge2mat(extward, num_node))
    A = np.stack((I, In, Out, Ext))
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_full_graph(num_node, self_link, edge):
    pass


class Graph:
    def __init__(self, strategy='spatial', layout='ntu-rgb+d'):
        self.A = self.get_adjacency_matrix(strategy)
        self.Ap = self.get_part_adjacency_matrix(strategy)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.inward_part = inward_part
        self.outward_part = outward_part
        self.neighbor_part = neighbor_part

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'ext':
            A = get_ext_graph(num_node, self_link, inward, outward, extward)
        elif labeling_mode == 'ones':
            A = get_full_graph(num_node, self_link, edge)
        else:
            raise ValueError()
        return A

    def get_part_adjacency_matrix(self, labeling_mode='spatial'):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial' or labeling_mode == 'ext' or labeling_mode == 'ones':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('ones').get_adjacency_matrix()
    Ap = Graph('spatial').get_part_adjacency_matrix()

    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()

    for j in Ap:
        plt.imshow(j, cmap='gray')
        plt.show()

    for i, j in zip(A, Ap):
        plt.imshow(i-j, cmap='gray')
        plt.show()


