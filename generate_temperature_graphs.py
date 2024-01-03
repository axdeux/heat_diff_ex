import torch
import numpy as np


def generate_temperature_graphs(graph, hotspots=None, temp_min_hotspot=0.5, temp_max_hotspot=1):


    data = graph.x.detach().numpy().copy()
    data_indices = np.arange(data.shape[0])
    if len(graph.dead_nodes) > 0:
        data_indices = np.delete(data_indices, graph.dead_nodes)
    
    if hotspots is None:
        num_hotspots = np.random.randint(int(data.shape[0]*0.01), int(data.shape[0]*0.75))
    else:
        num_hotspots = hotspots
    
    hotspot_indices = np.random.choice(data_indices, num_hotspots, replace=False)
    data[hotspot_indices] = np.random.uniform(temp_min_hotspot, temp_max_hotspot, num_hotspots)
    graph.x = torch.tensor(data)

    return graph


