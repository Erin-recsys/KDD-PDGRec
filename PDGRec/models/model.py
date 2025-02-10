import torch.nn as nn
from tqdm import tqdm
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F
from models.Social_Reweight import assign_weights_to_relationships
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class Proposed_model(nn.Module):
    def __init__(self, args, graph, graph_single, graph_or, graph_social,device, gamma=80, ablation=False):
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
        #self.gpu3 = torch.device("cuda:2")
        torch.cuda.empty_cache()#???
        self.args = args
        self.param_decay = args.param_decay
        self.hid_dim = args.embed_size  # default = 32
        self.attention_and = args.attention_and
        self.layer_num_and = args.layers_and
        self.layer_num_oricsr=3  
        self.layer_num_or = args.layers_or   
        self.layer_num_user_game = args.layers_user_game
        self.graph_single = graph_single.to(self.device_)
        self.graph_or = graph_or#.to(self.device_)
        self.graph = graph.to(self.device_)
        self.ori_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.ori_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)
        self.graph_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.graph_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)

        self.graph_social = graph_social

        path_weight_edge = base_dir + "/data_exist/weight_edge.pth"
        self.weight_edge = torch.load(path_weight_edge).to(self.device_)

        csr_path_weight_edge = base_dir + "/data_exist/csr_weight_edge.pth"
        self.csr_weight_edge = torch.load(csr_path_weight_edge).to(self.device_)

        path_weight_friend_of_DI = base_dir + "/data_exist/weight_friend_of_edge_DI.pth"
        self.weight_friend_of_DI = torch.load(path_weight_friend_of_DI).to(self.device_)
        self.weight_friend_of_DI = self.weight_friend_of_DI.to(torch.float32)

        path_weight_friend_of_CI = base_dir + "/data_exist/weight_friend_of_edge_CI.pth"
        self.weight_friend_of_CI = torch.load(path_weight_friend_of_CI).to(self.device_)
        self.weight_friend_of_CI = self.weight_friend_of_CI.to(torch.float32)

        csr_path_weight_friend_of_DI = base_dir + "/data_exist/csr_weight_friend_of_edge_DI.pth"
        self.csr_weight_friend_of_DI = torch.load(csr_path_weight_friend_of_DI).to(self.device_)
        self.csr_weight_friend_of_DI = self.csr_weight_friend_of_DI.to(torch.float32)

        csr_path_weight_friend_of_CI = base_dir + "/data_exist/csr_weight_friend_of_edge_CI.pth"
        self.csr_weight_friend_of_CI = torch.load(csr_path_weight_friend_of_CI).to(torch.float32)
        self.csr_weight_friend_of_CI = self.csr_weight_friend_of_CI.to(self.device_)

        path_weight_noise_edge = base_dir + "/data_exist/weight_noise_edge.pth"
        self.weight_noise_edge = torch.load(path_weight_noise_edge).to(self.device_)

        csr_path_weight_noise_edge = base_dir + "/data_exist/csr_weight_noise_edge.pth"
        self.csr_weight_noise_edge = torch.load(csr_path_weight_noise_edge).to(self.device_)


        self.edge_node_weight =True
        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('game').shape[0], self.hid_dim)).to(torch.float32)



        
        self.user_embed_social = torch.randn(self.graph.nodes('user').shape[0], self.hid_dim, requires_grad=False).to(torch.float32)
        self.user_embed_social = self.user_embed_social.to(self.device_)
        self.gamma = gamma
        self.w_or = self.gamma / (self.gamma + 2)
        self.w_and = self.w_or / self.gamma
        self.w_self = self.w_or / self.gamma


        self.W_and = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_and = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_or = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_or = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_social = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_social = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)

        self.conv1 = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True).to(self.device_)

        self.build_model_single(self.graph_single)
        self.build_model_ssl()
        self.build_model_csr()
        self.build_model_social()
        #self.build_model_or()
        #self.build_model_user_game()
        ###self.graph_contrast=self.Get_Contrast_views(self.graph,0.2)

        #self.graph_ori=dgl.edge_type_subgraph(self.graph, [('user', 'play', 'game'), ('game', 'played by', 'user')]).to(self.device_)
        #self.csr_user2item,self.csr_item2user=self.Get_Contrast_views(self.ori_item2user,self.ori_user2item,0.1)
        #self.csr_user2item,self.csr_item2user=self.csr_user2item.to(self.device_),self.csr_item2user.to(self.device_)
        #self.sub_model=MyGraphConvModel(self.hid_dim,self.graph_single, self.graph_ori, self.graph_csr, self.user_embedding,self.item_embedding,0.5,self.device_)

        contrast_graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "data_exist/contrast_graph.bin")
        if os.path.exists(contrast_graph_path):
            contrast_graph, _ = dgl.load_graphs(contrast_graph_path)
            self.graph_contrast = contrast_graph[0].to(self.device_)
            self.graph_contrast = dgl.edge_type_subgraph(self.graph_contrast, [('user','play','game'),('game','played by','user')])
            print("Successfully loaded contrast graph from file")
        else:
            print("No pre-computed contrast graph found, computing new one...")
            self.graph_contrast = self.Get_Contrast_views(self.graph, 0.5)
        self.graph_item2user_csr = dgl.edge_type_subgraph(self.graph_contrast,['played by']).to(self.device_)
        self.graph_user2item_csr = dgl.edge_type_subgraph(self.graph_contrast,['play']).to(self.device_)

        self.graph_social_csr = graph_social
    def build_model_single(self, graph_single):
        self.sub_g1 = dgl.edge_type_subgraph(graph_single,['co_genre']).to(self.device_)
        #self.sub_g2 = dgl.edge_type_subgraph(graph_single,['co_dev']).to(self.device_)
        #self.sub_g3 = dgl.edge_type_subgraph(graph_single,['co_pub']).to(self.device_)

    def get_h_single(self,ego_embeddings_sub):#,attention):

        ls = [ego_embeddings_sub.to(self.device_)]
        h1 = ego_embeddings_sub.to(self.device_)
        h2 = ego_embeddings_sub.to(self.device_)
        h3 = ego_embeddings_sub.to(self.device_)

        #print(self.graph_ori.num_nodes)

        for _ in range(self.layer_num_and):
            h1 = self.conv1(self.sub_g1, h1)
            h2 = self.conv1(self.sub_g2, h2)
            h3 = self.conv1(self.sub_g3, h3)
            #ls.append(h1)
            #ls.append(h2)
            #ls.append(h3)
        return ((h1+h2+h3)/3).cpu()


    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim=0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(weight*a, dim=0)
        tensor_layers = torch.sum(tensor_layers * weight, dim=0)
        return tensor_layers
    '''model for graph_and'''

    def build_model_and(self, graph_and):
        self.sub_g1 = dgl.edge_type_subgraph(graph_and,['co_genre_pub']).to(self.device_)
        self.sub_g2 = dgl.edge_type_subgraph(graph_and,['co_genre_dev']).to(self.device_)
        self.sub_g3 = dgl.edge_type_subgraph(graph_and,['co_dev_pub']).to(self.device_)


    def get_h_and(self,attention):
        ls = [self.item_embedding.to(self.device_)]
        h1 = self.item_embedding.to(self.device_)
        h2 = self.item_embedding.to(self.device_)
        h3 = self.item_embedding.to(self.device_)

        for _ in range(self.layer_num_and):
            h1 = self.conv1(self.sub_g1, h1)
            h2 = self.conv1(self.sub_g2, h2)
            h3 = self.conv1(self.sub_g3, h3)
            ls.append(h1)
            ls.append(h2)
            ls.append(h3)
        
        if attention == True:
            return self.layer_attention(ls, self.W_and.to(self.device_), self.a_and.to(self.device_))
        else:   
            return ((h1+h2+h3)/3).cpu()

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

    def get_h_oricsr(self, graph, h):
        for layer in self.layers:
            h = layer(graph, h)
        return h
    def build_model_social(self):
        self.layers_social = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer_social = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree=True)
            self.layers_social.append(layer_social)
        self.layers_social.to(self.device_)
           
    def forward(self):
        h = {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        for layer in self.layers:
            if self.edge_node_weight == True:
                h_user = layer(self.graph_item2user, (h['game'],h['user']))#,edge_weight=self.weight_edge)
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
                h_user = layer_csr(self.graph_item2user_csr, (h1['game'],h1['user']))#,edge_weight=self.csr_weight_edge)
                h_item = layer_csr(self.graph_user2item_csr, (h1['user'],h1['game']))
                h_noise=layer_csr(self.graph_item2user_csr, (h1['game'],h1['user']),edge_weight=self.csr_weight_noise_edge)

                h1_DI = layer_csr(self.graph_social_csr, (h1['user'], h1['user']),edge_weight=self.csr_weight_friend_of_DI)
                h1_CI = layer_csr(self.graph_social_csr, (h1['user'], h1['user']),edge_weight=self.csr_weight_friend_of_CI)

                h1['user'] = h_user*self.args.weight_self+h_noise*self.args.weight_noise + h1_DI*self.args.weight_DI+h1_CI*self.args.weight_CI
                h1['game'] = h_item
            else:
                h1 = layer(self.graph,h)

        return h,h_sub1,h1
    

    def Get_Contrast_views(self, graph, β):

        contrast_graph = graph.clone()

        play_percentiles = graph.edges['play'].data['percentile']
        played_by_percentiles = graph.edges['played by'].data['percentile']

        play_mask = play_percentiles >= β
        played_by_mask = played_by_percentiles >= β

        play_edges_to_remove = torch.nonzero(~play_mask, as_tuple=False).squeeze()
        played_by_edges_to_remove = torch.nonzero(~played_by_mask, as_tuple=False).squeeze()

        contrast_graph.remove_edges(play_edges_to_remove, etype='play')
        contrast_graph.remove_edges(played_by_edges_to_remove, etype='played by')

        return contrast_graph

class SSLoss():
    def __init__(self,args):
        super(SSLoss, self).__init__()
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = 1
        self.ssl_mode = 'both side'
        self.ssl_game_weight=args.ssl_game_weight


    def forward(self,ua_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub1,
                ia_embeddings_sub2,device):
        user_emb1 = ua_embeddings_sub1
        user_emb2 = ua_embeddings_sub2  # [B, dim]
        normalize_user_emb1 = F.normalize(user_emb1, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, dim=1)
        normalize_all_user_emb2 = F.normalize(ua_embeddings_sub2, dim=1)
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),
                                    dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)

        ttl_score_user = torch.matmul(normalize_user_emb1,
                                        normalize_all_user_emb2.T)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  # [B, ]

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
        #print(type(ssl_loss_item), type(ssl_loss_user), type(loss))
        return loss