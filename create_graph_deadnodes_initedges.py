import numpy as np
import torch
from generate_grid_graph import generate_grid_graph


def create_graph(N, M, dead_nodes=0, init_temp=10e-10):
    basic_grid_graph = generate_grid_graph(np.zeros((N, M))+init_temp)
    basic_grid_graph.dead_nodes = np.array([])
    basic_grid_graph.grid_mask = np.ones((N, M))*np.nan
    basic_grid_graph.grid_shape = (N, M)
    basic_grid_graph.num_nodes = int(N*M - dead_nodes)
    

    if dead_nodes > 0:
        dead_nodes = np.random.choice(N*M, dead_nodes, replace=False)
        # sort from largest to smallest
        dead_nodes = np.sort(dead_nodes)[::-1]
        
        # Remove dead nodes from the graph
        x_data = basic_grid_graph.x.detach().numpy()
        x_data[dead_nodes] = 0
        basic_grid_graph.x = torch.tensor(x_data)

        grid_mask = basic_grid_graph.grid_mask.flatten()
        grid_mask[dead_nodes] = 0
        grid_mask = grid_mask.reshape(N, M)
        basic_grid_graph.grid_mask = grid_mask
        
        # Remove any pair of edges that contain a dead node index
        edge_indices = basic_grid_graph.edge_index.t().numpy()

        
        for i in dead_nodes:
            edge_indices = edge_indices[~np.isin(edge_indices, i).any(axis=1)]

        ind_list = np.arange(N*M)
        ind_list = np.delete(ind_list, dead_nodes)

        for i in ind_list[::-1]:
            if not np.isin(i, edge_indices):
                dead_nodes = np.append(dead_nodes, i)

        basic_grid_graph.edge_index = torch.tensor(edge_indices).t().contiguous()
        basic_grid_graph.dead_nodes = dead_nodes
        basic_grid_graph.num_nodes = N*M - len(dead_nodes)
        
    return basic_grid_graph
    
