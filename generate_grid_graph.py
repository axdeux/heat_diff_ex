import networkx as nx
import torch
import torch_geometric as tg
import numpy as np
import matplotlib.pyplot as plt


def generate_grid_graph(grid):
    """Generates a grid graph from a grid matrix
    
    Args:
        grid (np.array): Grid matrix

    Returns:
        tg.data.Data: Graph data object
    """
    N, M = grid.shape
    #All nodes connect to its nearest neighbours (left, right, up, down)
    edge_index = []
    for i in range(N):
        for j in range(M):
            #Check if not on the right edge
            if j != M-1:
                edge_index.append([i*M+j, i*M+j+1])
            #Check if not on the bottom edge
            if i != N-1:
                edge_index.append([i*M+j, (i+1)*M+j])
            if i != 0:
                edge_index.append([i*M+j, (i-1)*M+j])
            if j != 0:
                edge_index.append([i*M+j, i*M+j-1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = tg.data.Data(x=torch.tensor(grid, dtype=torch.float).flatten(), edge_index=edge_index)
    return data

