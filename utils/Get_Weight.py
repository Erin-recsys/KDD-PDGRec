import dgl
import torch

def get_weight(fixed_value):
    """
    保存图中play边的各种权重
    """
    # 处理原始图
    print("Processing original graph...")
    path = "/home/zhangjingmao/data/PDGRec/data_exist/graph.bin"
    graph = dgl.load_graphs(path)[0]
    graph = graph[0]
    
    # 获取并处理play边权重
    edge_weights = graph.edges['play'].data['percentile']
    edge_weights = edge_weights + fixed_value
    
    # 获取friend of边权重
    friend_of_weights_DI = graph.edges['friend of'].data['DI']
    friend_of_weights_CI = graph.edges['friend of'].data['CI']

    # 加载带有noise weight的图
    print("\nLoading graph with noise weights...")
    noise_path = "/home/zhangjingmao/data/PDGRec/data_exist/graph_with_weights.bin"
    noise_graph = dgl.load_graphs(noise_path)[0]
    noise_graph = noise_graph[0]
    
    # 获取noise weight
    noise_weights = 1- noise_graph.edges['play'].data['noise_weight']

    #print(noise_weights)
    #import pdb; pdb.set_trace() 
    # 保存原始图的权重
    save_path = "/home/zhangjingmao/data/PDGRec/data_exist/"
    torch.save(edge_weights, save_path + "weight_edge.pth")
    torch.save(friend_of_weights_DI, save_path + "weight_friend_of_edge_DI.pth")
    torch.save(friend_of_weights_CI, save_path + "weight_friend_of_edge_CI.pth")
    torch.save(noise_weights, save_path + "weight_noise_edge.pth")

    # 处理对比图
    print("\nProcessing contrast graph...")
    path = "/home/zhangjingmao/data/PDGRec/data_exist/contrast_graph.bin"
    graph = dgl.load_graphs(path)[0]
    graph = graph[0]
        # 添加调试信息
    print("Contrast graph edge types:", graph.etypes)
    print("Contrast graph canonical edge types:", graph.canonical_etypes)
    # 获取并处理play边权重
    csr_edge_weights = graph.edges['play'].data['percentile']
    csr_edge_weights = csr_edge_weights + fixed_value
    
    # 获取friend of边权重
    csr_friend_of_weights_DI = graph.edges['friend of'].data['DI']
    csr_friend_of_weights_CI = graph.edges['friend of'].data['CI']

    # 加载对比图的noise weight
    print("\nLoading contrast graph with noise weights...")
    csr_noise_path = "/home/zhangjingmao/data/PDGRec/data_exist/contrast_graph_with_weights.bin"
    csr_noise_graph = dgl.load_graphs(csr_noise_path)[0]
    csr_noise_graph = csr_noise_graph[0]
    
    # 获取noise weight
    csr_noise_weights = 1 - csr_noise_graph.edges['play'].data['noise_weight']
    
    # 保存对比图的权重
    torch.save(csr_edge_weights, save_path + "csr_weight_edge.pth")
    torch.save(csr_friend_of_weights_DI, save_path + "csr_weight_friend_of_edge_DI.pth")
    torch.save(csr_friend_of_weights_CI, save_path + "csr_weight_friend_of_edge_CI.pth")
    torch.save(csr_noise_weights, save_path + "csr_weight_noise_edge.pth")

    # 打印统计信息
    print("\nWeight Statistics:")
    print("Original graph:")
    print(f"Play edges: count={len(edge_weights)}, mean={edge_weights.mean():.4f}")
    print(f"Friend of edges (DI): count={len(friend_of_weights_DI)}, mean={friend_of_weights_DI.mean():.4f}")
    print(f"Friend of edges (CI): count={len(friend_of_weights_CI)}, mean={friend_of_weights_CI.mean():.4f}")
    print(f"Noise weights: count={len(noise_weights)}, mean={noise_weights.mean():.4f}")
    
    print("\nContrast graph:")
    print(f"Play edges: count={len(csr_edge_weights)}, mean={csr_edge_weights.mean():.4f}")
    print(f"Friend of edges (DI): count={len(csr_friend_of_weights_DI)}, mean={csr_friend_of_weights_DI.mean():.4f}")
    print(f"Friend of edges (CI): count={len(csr_friend_of_weights_CI)}, mean={csr_friend_of_weights_CI.mean():.4f}")
    print(f"Noise weights: count={len(csr_noise_weights)}, mean={csr_noise_weights.mean():.4f}")
    
    print("\nWeights saved successfully!")

if __name__ == "__main__":
    get_weight(0)



'''import dgl
import torch

def get_weight(fixed_value):
    """
    保存图中play边的percentile权重和friend of边的DI权重
    """
    # 处理原始图
    print("Processing original graph...")
    path = "/home/zhangjingmao/data/PDGRec/data_exist/graph.bin"
    graph = dgl.load_graphs(path)[0]
    graph = graph[0]
    
    # 获取并处理play边权重
    edge_weights = graph.edges['play'].data['percentile']
    edge_weights = edge_weights + fixed_value
    
    # 获取friend of边权重
    friend_of_weights_DI = graph.edges['friend of'].data['DI']
    friend_of_weights_CI = graph.edges['friend of'].data['CI']
    # 保存原始图的权重
    save_path = "/home/zhangjingmao/data/PDGRec/data_exist/"
    torch.save(edge_weights, save_path + "weight_edge.pth")
    torch.save(friend_of_weights_DI, save_path + "weight_friend_of_edge_DI.pth")
    torch.save(friend_of_weights_CI, save_path + "weight_friend_of_edge_CI.pth")
    # 处理对比图
    print("\nProcessing contrast graph...")
    path = "/home/zhangjingmao/data/PDGRec/data_exist/contrast_graph.bin"
    graph = dgl.load_graphs(path)[0]
    graph = graph[0]
    
    # 获取并处理play边权重
    csr_edge_weights = graph.edges['play'].data['percentile']
    csr_edge_weights = csr_edge_weights + fixed_value
    
    # 获取friend of边权重
    csr_friend_of_weights_DI = graph.edges['friend of'].data['DI']
    csr_friend_of_weights_CI = graph.edges['friend of'].data['CI']
    # 保存对比图的权重
    torch.save(csr_edge_weights, save_path + "csr_weight_edge.pth")
    torch.save(csr_friend_of_weights_DI, save_path + "csr_weight_friend_of_edge_DI.pth")
    torch.save(csr_friend_of_weights_CI, save_path + "csr_weight_friend_of_edge_CI.pth")
    # 打印统计信息
    print("\nWeight Statistics:")
    print("Original graph:")
    print(f"Play edges: count={len(edge_weights)}, mean={edge_weights.mean():.4f}")
    print(f"Friend of edges: count={len(friend_of_weights_DI)}, mean={friend_of_weights_DI.mean():.4f}")
    print(f"Friend of edges: count={len(friend_of_weights_CI)}, mean={friend_of_weights_CI.mean():.4f}")
    
    print("\nContrast graph:")
    print(f"Play edges: count={len(csr_edge_weights)}, mean={csr_edge_weights.mean():.4f}")
    print(f"Friend of edges: count={len(csr_friend_of_weights_DI)}, mean={csr_friend_of_weights_DI.mean():.4f}")
    print(f"Friend of edges: count={len(csr_friend_of_weights_CI)}, mean={csr_friend_of_weights_CI.mean():.4f}")
    
    print("\nWeights saved successfully!")

if __name__ == "__main__":
    get_weight(0)'''