import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm ,GCNConv, global_mean_pool as gep
from core.edge_gcn import GCNEdgeConv
from torch_geometric.nn import global_max_pool
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from core.models.graph_transformer_pytorch import GraphTransformer
from torch_geometric.utils import to_dense_batch, to_dense_adj


class spatt(nn.Module):
    def __init__(self, padding = 3):
        super(spatt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(2*padding+1),padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xc = x.unsqueeze(1) #128,1,128
        avg = torch.mean(xc, dim=1, keepdim=True) 
        max_x, _ = torch.max(xc, dim=1, keepdim=True)
        xt = torch.cat((avg,max_x),dim=1) #64,2,128
        att = self.sigmoid(self.conv1(xt))  #64,2,128
        att_reshaped = att.view(128, 128)
        # print(att.shape) #128,2,128
        # print(x.shape) #128,128
        # exit()
        # print(att.squeeze(1).shape)
        return x * att_reshaped

#通道-SE
class SE_Block(nn.Module):                         # Squeeze-and-Excitation block()
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape) #128,256
        # exit()
        x = self.avgpool(x) #128,1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out
    
    
class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b


class AttentionMatrix(nn.Module):
    def __init__(self, input_dim=128):
        super(AttentionMatrix, self).__init__()

        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)

    def forward(self, xt, xg):

        Q = self.linear_q(xt)
        K = self.linear_k(xg)
        V = self.linear_v(xg)


        attention_scores = torch.matmul(Q, K.transpose(1, 0))


        attention_weights = F.softmax(attention_scores, dim=-1)


        attention_output = torch.matmul(attention_weights, V)


        attention_matrix = attention_weights.view(xt.size(0), 128, 128)

        return attention_matrix, attention_output
    
    
class self_attention(nn.Module):
    def __init__(self, channel):
        super(self_attention, self).__init__()
        self.linear_Q = nn.Linear(channel, channel)
        self.linear_K = nn.Linear(channel, channel)
        self.linear_V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.linear_Q(xs)
        K = self.linear_K(xs)
        V = self.linear_V(xs)
        scale = K.size(-1) ** -0.5
        att = self.softmax(Q * scale)
        ys = att * V
        return ys


class Attention1(nn.Module):
    def __init__(self, seq_len=128, d_k=128):

        super(Attention1, self).__init__()
        self.seq_len = seq_len
        self.d_k = d_k

        self.W_q = nn.Linear(1, d_k)
        self.W_k = nn.Linear(1, d_k)
        self.W_v = nn.Linear(1, d_k)

    def forward(self, xt, xg):

        xt_feat = xt.unsqueeze(-1)
        xg_feat = xg.unsqueeze(-1)

        Q = self.W_q(xt_feat)
        K = self.W_k(xg_feat)


        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))


        attn_matrix = F.softmax(scores, dim=-1)

        return attn_matrix

    


class CNN_MLP(nn.Module):
    def __init__(self, patch, channel, output_size, dr, down=False, last=False):
        super(CNN_MLP, self).__init__()
        self.Afine_p1 = Affine(channel)
        self.Afine_p2 = Affine(channel)
        self.Afine_p3 = Affine(channel)
        self.Afine_p4 = Affine(channel)
        self.linear0 = nn.Linear(patch, patch)
        self.linear1 = nn.Linear(1200, 1200)
        self.cross_patch_linear = nn.Linear(1200, patch)
        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=patch, kernel_size=15, padding=7, groups=patch)
        self.bn1 = nn.BatchNorm1d(patch)
        self.cnn2 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=31, padding=15, groups=patch)
        self.bn2 = nn.BatchNorm1d(patch)
        self.cnn3 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
        self.bn3 = nn.BatchNorm1d(patch)
        # self.self_attention = self_attention(channel)
        self.bnp1 = nn.BatchNorm1d(channel)

        self.atten = SE_Block(128)
        # self.att = spatt(3)
        # self.att_sp = spatial_attention(3)
        self.bnp = nn.BatchNorm1d(patch)
        self.act = nn.ReLU()
        self.last = last
        self.dropout = nn.Dropout(0.05)
        self.down = down

    def forward(self, x):

        x_cc1 = self.Afine_p3(x)
        x_cc1 = self.act(self.linear1(x_cc1))
        x_cc1 = self.Afine_p4(x_cc1)
        x_cc1 = self.act(self.linear1(x_cc1)) #3.25新加线性层)
        x_cc = x + x_cc1   #1,128,1000
        
        #测试提取模块
        # x = x.unsqueeze(0)
        # x = x.float() #1,128,1000
        # x_cc2 = self.act(self.bn1(self.cnn1(x)))

        x_cc2 = self.act(self.bn1(self.cnn1(x_cc)))
        x_cc2 = self.act(self.bn2(self.cnn2(x_cc2)))
        x_cc2 = self.act(self.bn3(self.cnn3(x_cc2)))
        # x_cc2 = self.Afine_p4(x_cc2)  #1,128,1000
        # x_cc2 = self.att_se(x_cc2)  #先concat再atten
        x_cc2 = self.atten(x_cc2) #1,128,1 
        # print(x_cc2.shape)
        # exit()
        
        x_out = x + self.dropout(x_cc2) #1,128,1000

        # if self.last == True:
        #  x_out = self.last_linear(x_out)
        return x_out


# GCN based model
class GCNEdgeNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=67, xg=1280, latent_dim=128, output_dim=128, dropout=0.2, edge_input_dim=None):
        super(GCNEdgeNet, self).__init__()

        self.n=0

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Initialize a list to store attention matrices W for each epoch
        self.saved_attention_matrices = []
        self.recorded_W = {}


        self.drug_branch = GraphTransformer(
            dim=128,
            depth=3,
            heads=8,
            with_feedforwards=True,
            accept_adjacency_matrix=True  # ★ 关键
        )



        # SMILES graph branch
        self.dconv1 = GCNEdgeConv(num_features_xd, num_features_xd, edge_input_dim=edge_input_dim, add_self_loops=False)
        self.dconv2 = GCNEdgeConv(num_features_xd, num_features_xd*2, edge_input_dim=edge_input_dim, add_self_loops=False)
        self.dconv3 = GCNEdgeConv(num_features_xd*2, num_features_xd * 4, edge_input_dim=edge_input_dim, add_self_loops=False)
        # self.dconv4 = GCNEdgeConv(num_features_xd*4, num_features_xd * 8, edge_input_dim=edge_input_dim, add_self_loops=False)
        # self.dconv5 = GCNEdgeConv(num_features_xd*8, num_features_xd * 16, edge_input_dim=edge_input_dim, add_self_loops=False)

        self.fc_gd1 = torch.nn.Linear(num_features_xd*4, 512)
        self.fc_gd2 = torch.nn.Linear(512, output_dim)

        # self.fc1_xd = nn.Linear(dim, output_dim)

        #  protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, latent_dim)
        self.cnnmlp = CNN_MLP(128, 1200, 128, 0, True)
        self.mlp_linear1 = torch.nn.Linear(1200, 1024)
        self.mlp_linear2 = torch.nn.Linear(1024, 512)
        self.mlp_linear3 = torch.nn.Linear(512, output_dim)

        #global feature
        self.glob_linear1 = torch.nn.Linear(xg, 1024)
        self.glob_linear2 = torch.nn.Linear(1024, 512)
        self.glob_linear3 = torch.nn.Linear(512, output_dim)

        #CNN combined
        self.atten = SE_Block(128)
        self.atten2 = spatt(3)
        self.atten3 = self_attention(128)
        self.atten1 = Attention1()
        self.cnn_linear = nn.Linear(output_dim, output_dim)

        self.attention_model = AttentionMatrix(input_dim=128)


        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.fc = nn.Linear(128, 128)

        # Norm layers
        self.DGnorm1 = GraphNorm(num_features_xd)
        self.DGnorm2 = GraphNorm(num_features_xd*2)
        self.DGnorm3 = GraphNorm(num_features_xd*4)
        self.DGnorm4 = GraphNorm(num_features_xd*8)
        self.DGnorm5 = GraphNorm(num_features_xd*16)
        self.TGnorm1 = GraphNorm(latent_dim)
        self.TGnorm2 = GraphNorm(latent_dim*2)
        self.TGnorm3 = GraphNorm(latent_dim*4)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(1024)
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.batchnorm = nn.BatchNorm1d(128)

    def forward(self, data_mol, data_prot):
        # get graph input
        x, edge_index, batch, edge_attr = data_mol.x, data_mol.edge_index, data_mol.batch, data_mol.edge_attr

        nodes, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch).squeeze(-1)

        x = self.dconv1(x, edge_index, edge_attr)  #x1: torch.Size([4157, 66])
        x = self.relu(self.DGnorm1(x))
        x = self.dconv2(x, edge_index,edge_attr)#x2: torch.Size([4157, 132])
        x = self.relu(self.DGnorm2(x))
        x = self.dconv3(x, edge_index,edge_attr) #x3: torch.Size([4157, 264])
        x = self.relu(self.DGnorm3(x))
        x = gep(x, batch) # global mean pooling 4157

        # flatten
        x = self.batchnorm1(self.fc_gd1(x))  #128,1024
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_gd2(x)
        # x = self.dropout(x)

        # target protein
        target = data_prot.target

        pocket = data_prot.pocket
        pocket = pocket.view(128, 128)


        cnnmlp = self.cnnmlp(target)  #1,128,1210
        xt = cnnmlp.view(128,1200)
        # print(xt.shape)
        xt = self.relu(self.mlp_linear1(xt))  #128，1024
        xt = self.relu(self.mlp_linear2(xt))
        xt = self.relu(self.mlp_linear3(xt))   #128,128
        # xt = self.dropout(xt)


        target_global = data_prot.x_global  #128,1,1280
        target_global = target_global.view(-1, 1280) #128,1280
        xg = self.glob_linear1(target_global)  #128,1024
        xg = self.batchnorm5(xg)
        xg = self.relu(xg)
        xg = self.glob_linear2(xg)
        xg = self.batchnorm6(xg)
        xg = self.relu(xg)
        xg = self.glob_linear3(xg) #128,128

        #fusion
        xp = xt + xg

        W = self.atten3(xp)
        Wa = self.atten1(xt,xg)
        # print("wa",  Wa.shape)
        Wout= xt *W+ xg*(1-W)#128,128

        for i in range(len(target)):
            attention_matrix = Wa[i].detach().cpu().numpy()

            self.saved_attention_matrices.append(attention_matrix)

        xc = torch.cat((x,Wout), 1)  #128,256
        # add some dense layers
        xc = self.batchnorm3(self.fc1(xc))
        # print("xc1:", xc.shape)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.batchnorm4(self.fc2(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)



        return out


