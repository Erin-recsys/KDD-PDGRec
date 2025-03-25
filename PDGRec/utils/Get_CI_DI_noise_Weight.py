import dgl
import torch

def get_weight(fixed_value):


    print("Processing original graph...")
    path = "./data_exist/graph.bin"
    graph = dgl.load_graphs(path)[0]
    graph = graph[0]
    

    edge_weights = graph.edges['play'].data['percentile']
    edge_weights = edge_weights + fixed_value
    

    friend_of_weights_DI = graph.edges['friend of'].data['DI']
    friend_of_weights_CI = graph.edges['friend of'].data['CI']


    print("\nLoading graph with noise weights...")
    noise_path = "./data_exist/graph_with_weights.bin"
    noise_graph = dgl.load_graphs(noise_path)[0]
    noise_graph = noise_graph[0]
    

    noise_weights = 1- noise_graph.edges['play'].data['noise_weight']


    save_path = "./data_exist/"
    torch.save(edge_weights, save_path + "weight_edge.pth")
    torch.save(friend_of_weights_DI, save_path + "weight_friend_of_edge_DI.pth")
    torch.save(friend_of_weights_CI, save_path + "weight_friend_of_edge_CI.pth")
    torch.save(noise_weights, save_path + "weight_noise_edge.pth")


    print("\nProcessing dn graph...")
    path = "./data_exist/dn_graph.bin"
    graph = dgl.load_graphs(path)[0]
    graph = graph[0]
  
    print("dn graph edge types:", graph.etypes)
    print("dn graph canonical edge types:", graph.canonical_etypes)
 
    dn_edge_weights = graph.edges['play'].data['percentile']
    dn_edge_weights = dn_edge_weights + fixed_value
    
    dn_friend_of_weights_DI = graph.edges['friend of'].data['DI']
    dn_friend_of_weights_CI = graph.edges['friend of'].data['CI']

 
    print("\nLoading dn graph with noise weights...")
    dn_noise_path = "./data_exist/dn_graph_with_weights.bin"
    dn_noise_graph = dgl.load_graphs(dn_noise_path)[0]
    dn_noise_graph = dn_noise_graph[0]
    
  
    dn_noise_weights = 1 - dn_noise_graph.edges['play'].data['noise_weight']
    
  
    torch.save(dn_edge_weights, save_path + "dn_weight_edge.pth")
    torch.save(dn_friend_of_weights_DI, save_path + "dn_weight_friend_of_edge_DI.pth")
    torch.save(dn_friend_of_weights_CI, save_path + "dn_weight_friend_of_edge_CI.pth")
    torch.save(dn_noise_weights, save_path + "dn_weight_noise_edge.pth")

 
    print("\nWeight Statistics:")
    print("Original graph:")
    print(f"Play edges: count={len(edge_weights)}, mean={edge_weights.mean():.4f}")
    print(f"Friend of edges (DI): count={len(friend_of_weights_DI)}, mean={friend_of_weights_DI.mean():.4f}")
    print(f"Friend of edges (CI): count={len(friend_of_weights_CI)}, mean={friend_of_weights_CI.mean():.4f}")
    print(f"Noise weights: count={len(noise_weights)}, mean={noise_weights.mean():.4f}")
    
    print("\ndn graph:")
    print(f"Play edges: count={len(dn_edge_weights)}, mean={dn_edge_weights.mean():.4f}")
    print(f"Friend of edges (DI): count={len(dn_friend_of_weights_DI)}, mean={dn_friend_of_weights_DI.mean():.4f}")
    print(f"Friend of edges (CI): count={len(dn_friend_of_weights_CI)}, mean={dn_friend_of_weights_CI.mean():.4f}")
    print(f"Noise weights: count={len(dn_noise_weights)}, mean={dn_noise_weights.mean():.4f}")
    
    print("\nWeights saved successfully!")

if __name__ == "__main__":
    get_weight(0)






