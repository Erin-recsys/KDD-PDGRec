import dgl
import torch
import os
from tqdm import tqdm
import numpy as np

def add_user_type_edges(graph):

   
    user_nodes, game_nodes = graph.edges(etype='play')
    play_times = graph.edges['play'].data['time']
    
   
    game_nodes_genre, type_nodes = graph.edges(etype='genre')
    
    
    user_type_stats = {}  # (user_id, type_id) -> (total_games, zero_time_games)
    
 
    
    for i in tqdm(range(len(user_nodes)), desc="Processing edges"):
        user_id = int(user_nodes[i])
        game_id = int(game_nodes[i])
        play_time = float(play_times[i])
        
       
        game_mask = (game_nodes_genre == game_id)
        game_types = type_nodes[game_mask]
        
      
        for type_id in game_types:
            type_id = int(type_id)
            key = (user_id, type_id)
            if key not in user_type_stats:
                user_type_stats[key] = [0, 0]  # [total_games, zero_time_games]
            user_type_stats[key][0] += 1
            if play_time == 0:
                user_type_stats[key][1] += 1


   
    user_weights = {}  # user_id -> {type_id: weight}
    for (user_id, type_id), stats in user_type_stats.items():
        if user_id not in user_weights:
            user_weights[user_id] = {}
        noise_ratio = stats[1] / stats[0]
        user_weights[user_id][type_id] = noise_ratio

 
    for user_id in user_weights:
        noise_ratios = list(user_weights[user_id].values())
        if len(noise_ratios) > 1: 
            min_ratio = min(noise_ratios)
            max_ratio = max(noise_ratios)
            if max_ratio != min_ratio: 
                for type_id in user_weights[user_id]:
                    normalized = (user_weights[user_id][type_id] - min_ratio) / (max_ratio - min_ratio)
                    user_weights[user_id][type_id] = 1 - normalized
            else:
              
                for type_id in user_weights[user_id]:
                    user_weights[user_id][type_id] = 1.0
        else: 
            type_id = list(user_weights[user_id].keys())[0]
            user_weights[user_id][type_id] = 1.0

  
    user_type_pairs = list(user_type_stats.keys())
    if not user_type_pairs:
        return graph
        

    src_nodes = torch.tensor([p[0] for p in user_type_pairs])
    dst_nodes = torch.tensor([p[1] for p in user_type_pairs])
    
  
    noise_ratios = torch.tensor([
        stats[1] / stats[0] for stats in user_type_stats.values()
    ], dtype=torch.float32)
    
   
    weights = torch.tensor([
        user_weights[p[0]][p[1]] for p in user_type_pairs
    ], dtype=torch.float32)

   
    new_graph_data = {
        ('game', 'genre', 'type'): graph.edges(etype='genre'),
        ('type', 'genred', 'game'): graph.edges(etype='genred'),
        ('user', 'play', 'game'): graph.edges(etype='play'),
        ('game', 'played by', 'user'): graph.edges(etype='played by'),
        ('user', 'interact', 'type'): (src_nodes, dst_nodes),
        ('type', 'interacted by', 'user'): (dst_nodes, src_nodes)
    }
    
   
    new_graph = dgl.heterograph(new_graph_data)
    
  
    new_graph.edges['play'].data['time'] = graph.edges['play'].data['time']
    new_graph.edges['played by'].data['time'] = graph.edges['played by'].data['time']
    new_graph.edges['play'].data['percentile'] = graph.edges['play'].data['percentile']
    new_graph.edges['played by'].data['percentile'] = graph.edges['played by'].data['percentile']
    
  
    new_graph.edges['interact'].data['noise_ratio'] = noise_ratios
    new_graph.edges['interacted by'].data['noise_ratio'] = noise_ratios
    
  
    new_graph.edges['interact'].data['weight'] = weights
    new_graph.edges['interacted by'].data['weight'] = weights
    
  
   
    play_weights = []
    for i in tqdm(range(len(user_nodes)), desc="Computing game weights"):
        user_id = int(user_nodes[i])
        game_id = int(game_nodes[i])
      
        game_mask = (game_nodes_genre == game_id)
        game_types = type_nodes[game_mask]
      
        game_weights = [user_weights[user_id][int(t)] for t in game_types]
        play_weights.append(float(np.mean(game_weights)))
    
    play_weights = torch.tensor(play_weights, dtype=torch.float32)
    new_graph.edges['play'].data['noise_weight'] = play_weights
    new_graph.edges['played by'].data['noise_weight'] = play_weights
    
    return new_graph

def process_and_save_graph(input_path, output_path):
   
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        print("Skipping graph processing...")
        return
        
    
    print(f"Processing graph from {input_path}")
    g = dgl.load_graphs(input_path)[0][0]
    
    new_g = add_user_type_edges(g)
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dgl.save_graphs(output_path, [new_g])
    print(f"Graph saved to {output_path}")
    

input_path = "./data_exist/graph.bin"
output_path = "./data_exist/graph_with_weights.bin"
process_and_save_graph(input_path, output_path)
dn_input_path = "./data_exist/dn_graph.bin"
dn_output_path = "./data_exist/dn_graph_with_weights.bin"
process_and_save_graph(dn_input_path, dn_output_path)
