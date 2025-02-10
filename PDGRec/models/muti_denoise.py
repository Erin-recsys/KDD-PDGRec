import torch
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
import torch.functional as F

class MyGraphConvModel(torch.nn.Module):
    def __init__(self,emb_size,graph_single,graph_ori,graph_csr,user_emb,item_emb,tau,device):
        super(MyGraphConvModel, self).__init__()
        self.graph_ori=graph_ori.to(device)
        self.device=device
        self.emb_size=emb_size
        self.graph_single=graph_single
        self.graph_csr=graph_csr.to(device)
        self.user_emb=user_emb
        self.tau=0.5
        self.item_emb=item_emb
        self.conv1 = GraphConv(self.emb_size, self.emb_size,allow_zero_in_degree=True).to(self.device)
        self.conv2 = GraphConv(self.emb_size, self.emb_size,allow_zero_in_degree=True).to(self.device)

        self.build_model_single(self.graph_single)


    def build_model_single(self, graph_single):
        self.sub_g1 = dgl.edge_type_subgraph(graph_single,['co_genre']).to(self.device)
        self.sub_g2 = dgl.edge_type_subgraph(graph_single,['co_dev']).to(self.device)
        self.sub_g3 = dgl.edge_type_subgraph(graph_single,['co_pub']).to(self.device)

    def layer_attention(self, ls, W, a):#(ls, self.W_and.to(self.device), self.a_and.to(self.device))
        tensor_layers = torch.stack(ls, dim=0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(weight*a, dim=0)
        tensor_layers = torch.sum(tensor_layers * weight, dim=0)
        return tensor_layers



    def get_h_single(self):
        ls = [self.item_embedding.to(self.device)]
        h1 = self.item_embedding.to(self.device)
        h2 = self.item_embedding.to(self.device)
        h3 = self.item_embedding.to(self.device)

        for _ in range(self.layer_num_and):
            h1 = self.conv1(self.sub_g1, h1)
            h2 = self.conv1(self.sub_g2, h2)
            h3 = self.conv1(self.sub_g3, h3)
        return h1,h2,h3

    def contrastive_loss(self, h1, h2, h3):

        def info_nce_loss(anchor, positive, temperature):
            pos_sim = torch.sum(anchor * positive, dim=-1)  # (N,)
            neg_sim = torch.mm(anchor, anchor.T)
            neg_sim.fill_diagonal_(-float('inf'))

            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N, 1 + N)
            logits /= temperature

            labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
            loss = F.cross_entropy(logits, labels)
            return loss

        loss_1_2 = info_nce_loss(h1, h2, self.temperature)
        loss_2_3 = info_nce_loss(h2, h3, self.temperature)
        loss_1_3 = info_nce_loss(h1, h3, self.temperature)

        total_loss = loss_1_2 + loss_2_3 + loss_1_3
        return total_loss


    def forward(self):
        user_embedding_ori = self.user_emb.to(self.device)
        game_embedding_ori = self.item_emb.to(self.device)
        '''print(f"user_embedding_ori device: {user_embedding_ori.device}")
        print(f"game_embedding_ori device: {game_embedding_ori.device}")
        print(f"graph_ori['play'] device: {self.graph_ori['play'].device}")
        print(f"graph_ori['played by'] device: {self.graph_ori['played by'].device}")
        print(f"conv1 weight device: {self.conv1.weight.device}")
        print(f"conv1 bias device: {self.conv1.bias.device}")'''

        h_ori_user = self.conv1(self.graph_ori['play'], user_embedding_ori)
        h_ori_game = self.conv1(self.graph_ori['played by'], game_embedding_ori) 

        user_embedding_contrast = self.user_emb.to(self.device)
        game_embedding_contrast = self.item_emb.to(self.device)

        h_contrast_user = self.conv2(self.graph_csr['play'], user_embedding_contrast)
        h_contrast_game = self.conv2(self.graph_csr['played by'], game_embedding_contrast)

        return h_ori_user, h_ori_game, h_contrast_user, h_contrast_game
    #tau==0.5
    def compute_infonce_loss(self, tau=0.5):
        user_embeddings_ori,_,user_embeddings_contrast,_=self.forward()
        sim_matrix = torch.cosine_similarity(user_embeddings_ori.unsqueeze(1), user_embeddings_contrast.unsqueeze(0),
                                         dim=-1)
        pos_sim = torch.diag(sim_matrix)

        sim_matrix = sim_matrix / tau

        pos_sim_exp = torch.exp(pos_sim / tau)
        total_sim_exp = torch.exp(sim_matrix).sum(dim=1)

        loss = -torch.log(pos_sim_exp / total_sim_exp).mean()

        return loss
    def all_loss(self):
        h1,h2,h3=self.get_h_single()
        return self.compute_infonce_loss()+self.contrastive_loss(h1,h2,h3)