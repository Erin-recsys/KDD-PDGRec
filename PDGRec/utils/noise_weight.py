import dgl
import torch
import os
from tqdm import tqdm
import numpy as np

def add_user_type_edges(graph):
    """
    为用户和游戏类型之间添加双向边，计算噪声占比和权重
    """
    # 获取所有用户和游戏的边
    user_nodes, game_nodes = graph.edges(etype='play')
    play_times = graph.edges['play'].data['time']
    
    # 获取游戏到类型的映射
    game_nodes_genre, type_nodes = graph.edges(etype='genre')
    
    # 创建字典存储每个用户对每种类型的游戏统计
    user_type_stats = {}  # (user_id, type_id) -> (total_games, zero_time_games)
    
    # 添加进度条
    print("处理用户-游戏交互...")
    for i in tqdm(range(len(user_nodes)), desc="Processing edges"):
        user_id = int(user_nodes[i])
        game_id = int(game_nodes[i])
        play_time = float(play_times[i])
        
        # 找到这个游戏对应的所有类型
        game_mask = (game_nodes_genre == game_id)
        game_types = type_nodes[game_mask]
        
        # 为每个类型更新统计
        for type_id in game_types:
            type_id = int(type_id)
            key = (user_id, type_id)
            if key not in user_type_stats:
                user_type_stats[key] = [0, 0]  # [total_games, zero_time_games]
            user_type_stats[key][0] += 1
            if play_time == 0:
                user_type_stats[key][1] += 1

    print("计算噪声占比和权重...")
    # 按用户分组计算权重
    user_weights = {}  # user_id -> {type_id: weight}
    for (user_id, type_id), stats in user_type_stats.items():
        if user_id not in user_weights:
            user_weights[user_id] = {}
        noise_ratio = stats[1] / stats[0]
        user_weights[user_id][type_id] = noise_ratio

    # 对每个用户进行min-max归一化并计算权重
    for user_id in user_weights:
        noise_ratios = list(user_weights[user_id].values())
        if len(noise_ratios) > 1:  # 如果用户有多个类型
            min_ratio = min(noise_ratios)
            max_ratio = max(noise_ratios)
            if max_ratio != min_ratio:  # 避免除以零
                for type_id in user_weights[user_id]:
                    normalized = (user_weights[user_id][type_id] - min_ratio) / (max_ratio - min_ratio)
                    user_weights[user_id][type_id] = 1 - normalized
            else:
                # 如果所有值相同，设置为1
                for type_id in user_weights[user_id]:
                    user_weights[user_id][type_id] = 1.0
        else:  # 如果用户只有一个类型
            type_id = list(user_weights[user_id].keys())[0]
            user_weights[user_id][type_id] = 1.0

    # 准备新边的数据
    user_type_pairs = list(user_type_stats.keys())
    if not user_type_pairs:
        return graph
        
    print("生成新的边数据...")
    src_nodes = torch.tensor([p[0] for p in user_type_pairs])
    dst_nodes = torch.tensor([p[1] for p in user_type_pairs])
    
    # 计算噪声占比
    noise_ratios = torch.tensor([
        stats[1] / stats[0] for stats in user_type_stats.values()
    ], dtype=torch.float32)
    
    # 计算权重
    weights = torch.tensor([
        user_weights[p[0]][p[1]] for p in user_type_pairs
    ], dtype=torch.float32)

    # 创建新的图数据字典
    new_graph_data = {
        ('game', 'genre', 'type'): graph.edges(etype='genre'),
        ('type', 'genred', 'game'): graph.edges(etype='genred'),
        ('user', 'play', 'game'): graph.edges(etype='play'),
        ('game', 'played by', 'user'): graph.edges(etype='played by'),
        ('user', 'interact', 'type'): (src_nodes, dst_nodes),
        ('type', 'interacted by', 'user'): (dst_nodes, src_nodes)
    }
    
    print("创建新图...")
    new_graph = dgl.heterograph(new_graph_data)
    
    print("复制和添加边属性...")
    # 复制原有边的属性
    new_graph.edges['play'].data['time'] = graph.edges['play'].data['time']
    new_graph.edges['played by'].data['time'] = graph.edges['played by'].data['time']
    new_graph.edges['play'].data['percentile'] = graph.edges['play'].data['percentile']
    new_graph.edges['played by'].data['percentile'] = graph.edges['played by'].data['percentile']
    
    # 添加噪声占比属性
    new_graph.edges['interact'].data['noise_ratio'] = noise_ratios
    new_graph.edges['interacted by'].data['noise_ratio'] = noise_ratios
    
    # 添加权重属性
    new_graph.edges['interact'].data['weight'] = weights
    new_graph.edges['interacted by'].data['weight'] = weights
    
    # 为play和played by边添加noise_weight
    print("计算游戏的噪声权重...")
    play_weights = []
    for i in tqdm(range(len(user_nodes)), desc="Computing game weights"):
        user_id = int(user_nodes[i])
        game_id = int(game_nodes[i])
        # 找到这个游戏的所有类型
        game_mask = (game_nodes_genre == game_id)
        game_types = type_nodes[game_mask]
        # 获取该用户对这些类型的权重的平均值
        game_weights = [user_weights[user_id][int(t)] for t in game_types]
        play_weights.append(float(np.mean(game_weights)))
    
    play_weights = torch.tensor(play_weights, dtype=torch.float32)
    new_graph.edges['play'].data['noise_weight'] = play_weights
    new_graph.edges['played by'].data['noise_weight'] = play_weights
    
    return new_graph

# 使用示例:
def process_and_save_graph(input_path, output_path):
    """
    读取图，添加用户-类型双向边，并保存
    """
    print(f"正在加载图: {input_path}")
    g = dgl.load_graphs(input_path)[0][0]
    
    print("处理图...")
    new_g = add_user_type_edges(g)
    
    print(f"保存新图到: {output_path}")
    dgl.save_graphs(output_path, [new_g])
    
    # 打印统计信息
    print("\n统计信息:")
    print("-" * 50)
    print(f"原图边类型: {g.etypes}")
    print(f"新图边类型: {new_g.etypes}")
    print(f"新增 user-type 边数量: {new_g.num_edges('interact')}")
    
    # 权重统计
    weights = new_g.edges['interact'].data['weight']
    print(f"\n权重统计:")
    print(f"平均权重: {float(torch.mean(weights)):.3f}")
    print(f"最大权重: {float(torch.max(weights)):.3f}")
    print(f"最小权重: {float(torch.min(weights)):.3f}")
    
    play_weights = new_g.edges['play'].data['noise_weight']
    print(f"\n游戏噪声权重统计:")
    print(f"平均权重: {float(torch.mean(play_weights)):.3f}")
    print(f"最大权重: {float(torch.max(play_weights)):.3f}")
    print(f"最小权重: {float(torch.min(play_weights)):.3f}")
    print("-" * 50)
    print("\n处理完成!")

# 使用示例:
input_path = "/home/zhangjingmao/data/PDGRec/data_exist/graph.bin"
output_path = "/home/zhangjingmao/data/PDGRec/data_exist/graph_with_weights.bin"
process_and_save_graph(input_path, output_path)