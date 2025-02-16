import torch.nn as nn
from tqdm import tqdm
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F

import os
class Proposed_model(nn.Module):
    def __init__(self, args, graph, graph_item, graph_social,device, gamma=80, ablation=False):
        super().__init__()
        print("\n=== Graph Information ===")
        print("Node types:", graph.ntypes)  
        print("Edge types:", graph.etypes)  
        print("Canonical edge types:", graph.canonical_etypes)  
        print("\n=== Node Statistics ===")
        for ntype in graph.ntypes:
            print(f"Number of {ntype} nodes:", graph.number_of_nodes(ntype))
        print("\n=== Edge Statistics ===")
        for etype in graph.etypes:
            print(f"Number of {etype} edges:", graph.number_of_edges(etype))
        self.ablation = ablation
        self.device_ = torch.device(device)
        torch.cuda.empty_cache()
        self.args = args
        self.param_decay = args.param_decay
        self.hid_dim = args.embed_size  
        self.attention_and = args.attention_and
        self.layer_num_and = args.layers_and
        self.layer_num_oricsr=3  
        self.layer_num_or = args.layers_or   
        self.layer_num_user_game = args.layers_user_game
        self.graph_item = graph_item.to(self.device_)

        self.graph = graph.to(self.device_)
        self.ori_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.ori_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)
        self.graph_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.graph_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)

        self.graph_social = graph_social

        path_weight_edge = "/PDGRec/data_exist/weight_edge.pth"
        self.weight_edge = torch.load(path_weight_edge).to(self.device_)

        csr_path_weight_edge = "/PDGRec/data_exist/csr_weight_edge.pth"
        self.csr_weight_edge = torch.load(csr_path_weight_edge).to(self.device_)

        path_weight_friend_of_DI = "/PDGRec/data_exist/weight_friend_of_edge_DI.pth"
        self.weight_friend_of_DI = torch.load(path_weight_friend_of_DI).to(self.device_)
        self.weight_friend_of_DI = self.weight_friend_of_DI.to(torch.float32)

        path_weight_friend_of_CI = "/PDGRec/data_exist/weight_friend_of_edge_CI.pth"
        self.weight_friend_of_CI = torch.load(path_weight_friend_of_CI).to(self.device_)
        self.weight_friend_of_CI = self.weight_friend_of_CI.to(torch.float32)

        csr_path_weight_friend_of_DI = "/PDGRec/data_exist/csr_weight_friend_of_edge_DI.pth"
        self.csr_weight_friend_of_DI = torch.load(csr_path_weight_friend_of_DI).to(self.device_)
        self.csr_weight_friend_of_DI = self.csr_weight_friend_of_DI.to(torch.float32)

        csr_path_weight_friend_of_CI = "/PDGRec/data_exist/csr_weight_friend_of_edge_CI.pth"
        self.csr_weight_friend_of_CI = torch.load(csr_path_weight_friend_of_CI).to(torch.float32)
        self.csr_weight_friend_of_CI = self.csr_weight_friend_of_CI.to(self.device_)

        path_weight_noise_edge = "/PDGRec/data_exist/weight_noise_edge.pth"
        self.weight_noise_edge = torch.load(path_weight_noise_edge).to(self.device_)

        csr_path_weight_noise_edge = "/PDGRec/data_exist/csr_weight_noise_edge.pth"
        self.csr_weight_noise_edge = torch.load(csr_path_weight_noise_edge).to(self.device_)


        self.edge_node_weight =True
        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('game').shape[0], self.hid_dim)).to(torch.float32)



        
        self.user_embed_social = torch.randn(self.graph.nodes('user').shape[0], self.hid_dim, requires_grad=False).to(torch.float32)
        self.user_embed_social = self.user_embed_social.to(self.device_)



        self.W_and = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_and = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_or = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_or = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_social = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_social = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)

        self.conv1 = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True).to(self.device_)

        self.build_model_item(self.graph_item)
        self.build_model_ssl()
        self.build_model_csr()


        contrast_graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "data_exist/contrast_graph.bin")

        contrast_graph, _ = dgl.load_graphs(contrast_graph_path)
        self.graph_contrast = contrast_graph[0].to(self.device_)
        self.graph_contrast = dgl.edge_type_subgraph(self.graph_contrast, [('user','play','game'),('game','played by','user')])
        print("Successfully loaded contrast graph from file")

        self.graph_item2user_csr = dgl.edge_type_subgraph(self.graph_contrast,['played by']).to(self.device_)
        self.graph_user2item_csr = dgl.edge_type_subgraph(self.graph_contrast,['play']).to(self.device_)

        self.graph_social_csr = graph_social
    def build_model_item(self, graph_item):
        self.sub_g1 = dgl.edge_type_subgraph(graph_item,['co_genre']).to(self.device_)


    def build_model_ssl(self):
        self.layers = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer = 0
            if self.edge_node_weight == True:
                layer = GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
            else:
                layer = dgl.nn.HeteroGraphConv({
                    'play': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True),
                    'played by': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
                })
            self.layers.append(layer)
        self.layers.to(self.device_)
    def build_model_csr(self):
        self.layers_csr = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer_csr = 0
            if self.edge_node_weight == True:
                layer_csr = GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
            else:
                layer_csr = dgl.nn.HeteroGraphConv({
                    'play': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True),
                    'played by': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
                })
            self.layers_csr.append(layer_csr)
        self.layers_csr.to(self.device_)



           
    def forward(self):
        h = {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        for layer in self.layers:
            if self.edge_node_weight == True:
                h_user = layer(self.graph_item2user, (h['game'],h['user']))
                h_item = layer(self.graph_user2item, (h['user'],h['game']))
                h_noise=layer(self.graph_item2user, (h['game'],h['user']),edge_weight=self.weight_noise_edge)

                h_DI = layer(self.graph_social, (h['user'], h['user']),edge_weight=self.weight_friend_of_DI)
                h_CI = layer(self.graph_social, (h['user'], h['user']),edge_weight=self.weight_friend_of_CI)

                h['user'] = h_user*self.args.weight_self+h_noise*self.args.weight_noise + h_DI*self.args.weight_DI + h_CI*self.args.weight_CI  
                h['game'] = h_item
            else:
                h = layer(self.graph,h)
        h_sub1 = h
        h1= {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        for layer_csr in self.layers_csr:
            if self.edge_node_weight == True:
                h_user = layer_csr(self.graph_item2user_csr, (h1['game'],h1['user']))
                h_item = layer_csr(self.graph_user2item_csr, (h1['user'],h1['game']))
                h_noise=layer_csr(self.graph_item2user_csr, (h1['game'],h1['user']),edge_weight=self.csr_weight_noise_edge)

                h1_DI = layer_csr(self.graph_social_csr, (h1['user'], h1['user']),edge_weight=self.csr_weight_friend_of_DI)
                h1_CI = layer_csr(self.graph_social_csr, (h1['user'], h1['user']),edge_weight=self.csr_weight_friend_of_CI)

                h1['user'] = h_user*self.args.weight_self+h_noise*self.args.weight_noise + h1_DI*self.args.weight_DI+h1_CI*self.args.weight_CI  
                h1['game'] = h_item
            else:
                h1 = layer(self.graph,h)
        
        return h,h_sub1,h1
    


class SSLoss():
    def __init__(self,args):
        super(SSLoss, self).__init__()
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = 1
        self.ssl_game_weight=args.ssl_game_weight


    def forward(self,ua_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub1,
                ia_embeddings_sub2,device):
        user_emb1 = ua_embeddings_sub1
        user_emb2 = ua_embeddings_sub2  
        normalize_user_emb1 = F.normalize(user_emb1, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, dim=1)
        normalize_all_user_emb2 = F.normalize(ua_embeddings_sub2, dim=1)
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),
                                    dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)

        ttl_score_user = torch.matmul(normalize_user_emb1,
                                        normalize_all_user_emb2.T)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  

        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        item_emb1 = ia_embeddings_sub1
        item_emb2 = ia_embeddings_sub2

        normalize_item_emb1 = F.normalize(item_emb1, dim=1)
        normalize_item_emb2 = F.normalize(item_emb2, dim=1)
        normalize_all_item_emb2 = F.normalize(ia_embeddings_sub2, dim=1)
        pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1)
        ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.T)

        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)

        ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))*self.ssl_game_weight

        loss=ssl_loss_item+ssl_loss_user
        return loss
