import numpy as np
from tqdm import tqdm
from generate_temperature_graphs import generate_temperature_graphs
import os
from make_gif import make_gif
from create_graph_deadnodes_initedges import create_graph
from PIL import Image
import torch
import parameters as par

def get_laplacian(graph):
    # Get the adjecency matrix from edges
    adjecency_matrix = np.zeros((graph.grid_shape[0]*graph.grid_shape[1], graph.grid_shape[0]*graph.grid_shape[1]))
    for edge in graph.edge_index.T:
        adjecency_matrix[edge[0], edge[1]] = 1
        adjecency_matrix[edge[1], edge[0]] = 1
    
    # Get the degree matrix from adjecency matrix
    degree_matrix = np.zeros((graph.grid_shape[0]*graph.grid_shape[1], graph.grid_shape[0]*graph.grid_shape[1]))
    for i in range(adjecency_matrix.shape[0]):
        degree_matrix[i, i] = np.sum(adjecency_matrix[i, :])
    
    laplacian = adjecency_matrix - degree_matrix    #Calculate the laplacian operator

    return laplacian


def heat_diffusion(graph, timesteps=20, dt=0.1, alpha=0.5, save_path=None, save_name=None, gif=True, heatmap=None):
    laplacian = get_laplacian(graph)
    temperature_matrix = graph.x.numpy().copy()
    

    matrix_list = np.zeros((timesteps, graph.grid_shape[0], graph.grid_shape[1]))   #List of matrices for each timestep
    matrix_list[0] = temperature_matrix.reshape(graph.grid_shape)
    
    gamma = alpha*dt    #Diffusivity constant combined with timestep

    
    print("Calculating heat diffusion...")
    for t in tqdm(range(1, timesteps)):
        # Heat diffusion
        temperature_matrix = temperature_matrix + gamma*laplacian.dot(temperature_matrix)
        if heatmap is not None:
                temperature_matrix[heatmap[0]] = heatmap[1]
        matrix_list[t] = temperature_matrix.reshape(graph.grid_shape)

    #extra step is to store graphs for each timestep
        
    make_gif(matrix_list, graph.grid_mask, min_temp=par.min_gif_cmap_temp, max_temp=par.max_gif_cmap_temp, path=save_path, save_name=save_name)    #Make gif of the diffusion


def create_heat_graph(image, heatmap=False):
    im_frame = Image.open(image+'.png')
    np_frame = np.array(im_frame.getdata())[:, 0]
    np_frame = np_frame/255
    np_frame[np_frame == 1] = np.nan

    

    # reshape
    N, M = 100, 100
    np_frame = np_frame.reshape(N, M)

    # change dead_nodes to indices in grid
    dead_nodes = np.where(np_frame == 0)
    dead_nodes = np.array([dead_nodes[0], dead_nodes[1]]).T

    # flatten dead_nodes indices
    dead_nodes_int = np.ravel_multi_index(dead_nodes.T, (N, M))

    init_graph = create_graph(N, M, dead_nodes=0, init_temp=par.base_graph_temp)
    edge_indices = init_graph.edge_index.numpy().copy().T

    for i in dead_nodes_int:
        edge_indices = edge_indices[~np.isin(edge_indices, i).any(axis=1)]

    ind_list = np.arange(N*M)
    ind_list = np.delete(ind_list, dead_nodes_int)

    for i in ind_list[::-1]:
        if not np.isin(i, edge_indices):
            dead_nodes_int = np.append(dead_nodes_int, i)

    init_graph.edge_index = torch.tensor(edge_indices).t().contiguous()
    init_graph.dead_nodes = dead_nodes
    init_graph.grid_mask = np_frame

    temperature_grid = np.zeros((N, M))
    
    heat_source = None
    if heatmap:
        temperature_grid = temperature_grid.flatten()
        heat_frame = Image.open(image+'_heat.png')
        heat_frame = np.array(heat_frame.getdata())[:, 0]
        heat_frame = heat_frame/255
        heat_source = np.where(heat_frame == 0)[0]

        temperature_grid[heat_source] = par.hotspot_temp
    else:
        #custom heat source
        for i in range(40, 46):
            for j in range(50, 61):
                temperature_grid[i, j] = par.hotspot_temp
        temperature_grid = temperature_grid.flatten()
    
    
    init_graph.x = torch.tensor(temperature_grid).float()

    return init_graph, heat_source



heat_graph, heat_source = create_heat_graph(par.image_name, heatmap=par.heatmap)

if par.continous_heat:
    heat_diffusion(heat_graph, timesteps=par.total_time, dt=par.dt, alpha=par.alpha, save_path="gifs/", save_name=par.image_name, heatmap=(heat_source, par.hotspot_temp))
else:
    heat_diffusion(heat_graph, timesteps=par.total_time, dt=par.dt, alpha=par.alpha, save_path="gifs/", save_name=par.image_name)