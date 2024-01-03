import numpy as np
import torch
from create_graph_wdead import create_graph
import os
from make_gif import make_gif


def get_adjacency_matrix(graph):
    """Returns the adjacency matrix of a graph

    Args:
        graph (tg.data.Data): Graph data object

    Returns:
        np.array: Adjacency matrix
        np.array: Degree matrix
    """
    adjecency_matrix = np.zeros((graph.grid_shape[0]*graph.grid_shape[1], graph.grid_shape[0]*graph.grid_shape[1]))
    for edge in graph.edge_index.T:
        adjecency_matrix[edge[0], edge[1]] = 1
        adjecency_matrix[edge[1], edge[0]] = 1
    
    # Get the degree matrix from adjecency matrix
    degree_matrix = np.zeros((graph.grid_shape[0]*graph.grid_shape[1], graph.grid_shape[0]*graph.grid_shape[1]))
    for i in range(adjecency_matrix.shape[0]):
        degree_matrix[i, i] = np.sum(adjecency_matrix[i, :])
        
    return adjecency_matrix, degree_matrix


def get_laplacian_matrix(graph):
    """Returns the laplacian matrix of a graph

    Args:
        graph (tg.data.Data): Graph data object

    Returns:
        np.array: Laplacian matrix
    """
    adjecency_matrix, degree_matrix = get_adjacency_matrix(graph)
    laplacian_matrix = adjecency_matrix - degree_matrix
    return laplacian_matrix



def wave_calculation(graph, timesteps=20, dt = 0.2, wave_speed=1, save_path=None, save_name=None, gif=False):
    """Calculates the wave propagation on a graph

    Args:
        graph (tg.data.Data): Graph data object
        timesteps (int, optional): Number of timesteps. Defaults to 20.
        dt (float, optional): Timestep size. Defaults to 0.1.
        wave_speed (float, optional): Wave speed. Defaults to 0.2.

    Returns:
        np.array: Wave propagation data
    """
    timesteps += 2
    laplace = get_laplacian_matrix(graph)

    initial_wave = graph.x.detach().numpy().copy()
    second_initial_wave = initial_wave.copy()

    matrix_list = np.zeros((timesteps, graph.grid_shape[0], graph.grid_shape[1]))
    matrix_list[0] = initial_wave.reshape(graph.grid_shape)
    matrix_list[1] = second_initial_wave.reshape(graph.grid_shape)
    
    if save_path is not None:
        # Create the directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # print("Calculating wave propagation...")
    for t in range(2, timesteps):
        prev_wave = matrix_list[t-2].flatten()
        current_wave = matrix_list[t-1].flatten()

        # Calculate the output of the wave equation
        next_step = (wave_speed*dt)**2 * laplace.dot(current_wave) + 2*current_wave - prev_wave
        

        matrix_list[t] = next_step.reshape(graph.grid_shape)
    # print("-----------------------------")
    # print("Storing graphs...")
    for time in range(2, timesteps):
        prev = matrix_list[time-2].flatten()
        curr = matrix_list[time-1].flatten()
        next = matrix_list[time].flatten()
        stacked = np.stack((prev, curr), axis=1)
        stacked = torch.from_numpy(stacked).float()

        timestep_graph = create_graph(graph.grid_shape[0], graph.grid_shape[1])
        timestep_graph.x = stacked
        timestep_graph.y = torch.from_numpy(next).float()
        timestep_graph.edge_index = graph.edge_index
        timestep_graph.dead_nodes = graph.dead_nodes
        timestep_graph.grid_mask = graph.grid_mask
        timestep_graph.grid_shape = graph.grid_shape

        if save_name == None:
            torch.save(timestep_graph, os.path.join(save_path, "wave_{}.pt".format(time-2)))
        else:
            torch.save(timestep_graph, os.path.join(save_path, save_name+"_"+str(time-2)+".pt"))
    # print("-----------------------------")
    if gif:
        # print("Creating gif...")
        make_gif(matrix_list, timestep_graph.grid_mask, max_val=2, path=save_path, save_name=save_name)
    return None
