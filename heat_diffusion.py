import torch
import numpy as np
from generate_temperature_graphs import generate_temperature_graphs
import os
from make_gif import make_gif
from create_graph_deadnodes_initedges import create_graph
from tqdm import tqdm

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


def heat_diffusion(graph, timesteps=20, dt=0.1, alpha=0.5, save_path=None, save_name=None, gif=True):
    laplacian = get_laplacian(graph)
    temperature_matrix = graph.x.numpy().copy()
    

    matrix_list = np.zeros((timesteps, graph.grid_shape[0], graph.grid_shape[1]))   #List of matrices for each timestep
    matrix_list[0] = temperature_matrix.reshape(graph.grid_shape)
    
    gamma = alpha*dt    #Diffusivity constant combined with timestep

    
    if save_path is not None:
        # Create the directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    print("Calculating heat diffusion...")
    for t in tqdm(range(1, timesteps)):
        # Heat diffusion
        temperature_matrix = temperature_matrix + gamma*laplacian.dot(temperature_matrix)
        matrix_list[t] = temperature_matrix.reshape(graph.grid_shape)
    
    
    print("Saving graph data objects...")
    for time in range(1, timesteps-1):
        # Create the graph data object for each timestep containing the t-1 and t diffusion timesteps
        # and the t+1 timestep as the target
        prev = matrix_list[time-1].flatten()    
        curr = matrix_list[time].flatten()
        next = matrix_list[time+1].flatten()
        stacked = np.stack((prev, curr), axis=1)
        stacked = torch.from_numpy(stacked).float()

        timestep_graph = create_graph(graph.grid_shape[0], graph.grid_shape[1])
        timestep_graph.x = stacked  #Stack the previous and current timestep as the x data
        timestep_graph.y = torch.from_numpy(next).float()   #The target is the next timestep

        # Set same parameters as the original graph to keep the same structure
        timestep_graph.edge_index = graph.edge_index
        timestep_graph.dead_nodes = graph.dead_nodes
        timestep_graph.grid_mask = graph.grid_mask
        timestep_graph.grid_shape = graph.grid_shape

        if save_name == None:   #Save the graph data object
            torch.save(timestep_graph, os.path.join(save_path, "temp_{}.pt".format(time)))
        else:
            torch.save(timestep_graph, os.path.join(save_path, save_name+"_"+str(time)+".pt"))

    if gif:
        make_gif(matrix_list, timestep_graph.grid_mask, max_temp=1, path=save_path, save_name=save_name)    #Make gif of the diffusion
    return None


def generate(N, M, graph_amount, dead_nodes=0, time_steps=20, rand_hot=None, save_gif=True):
    """Generate a heat diffusion graph where each timestep is a graph data object.

    Args:
        N (int): Number of rows in the grid.
        M (int): Number of columns in the grid.
        graph_amount (int): Number of graphs to generate.
        dead_nodes (int, optional): Number of disconnected vertices in the grid. Defaults to 0.
        time_steps (int, optional): Number of timesteps in the heat diffusion. Defaults to 20.
        rand_hot (int, optional): Number of hotspots to generate. Defaults to None.
        save_gif (bool, optional): Whether to save a gif of the heat diffusion. Defaults to True.
    
    Returns:
        None
    """
    init_graph = create_graph(N, M, dead_nodes=dead_nodes)
    the_dead_nodes = init_graph.dead_nodes
    the_edge_indices = init_graph.edge_index
    the_grid_mask = init_graph.grid_mask
    the_num_nodes = init_graph.num_nodes

    for i in range(graph_amount):
        if rand_hot == None:    #If no random hotspots are given, generate a random amount
            rand_hot = np.random.randint(1, int((N*M - dead_nodes)*0.9))

        graph = create_graph(N, M, dead_nodes=dead_nodes)
        # Set same parameters as the original graph to keep the same structure
        graph.dead_nodes = the_dead_nodes   
        graph.edge_index = the_edge_indices
        graph.grid_mask = the_grid_mask
        graph.edge_index = the_edge_indices
        graph.num_nodes = the_num_nodes
        heat_graph = generate_temperature_graphs(graph, hotspots=rand_hot, temp_max_hotspot=20)

        heat_diffusion(heat_graph, timesteps=time_steps, save_path="long_time_{}/".format(dead_nodes), save_name=f"long_time_graph_{rand_hot}hot", gif=save_gif)
    


np.random.seed(0)

generate(150, 150, graph_amount=1, dead_nodes=1000, time_steps=250, rand_hot=100, save_gif=True)
