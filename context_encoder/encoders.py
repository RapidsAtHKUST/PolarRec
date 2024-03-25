
import os
import sys
import copy
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATConv, GATv2Conv
from torch import nn, einsum 
from einops import rearrange, repeat
# from torch_geometric.utils import dense_to_sparse, to_dense_adj, add_remaining_self_loops
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., adj_dim = 83):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        
        # self.adj_dim = adj_dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # self.adj_head = nn.Linear(adj_dim, adj_dim * heads) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads # n- # of tokens, h- # of heads, n- # of dimensions for each head
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # adj = self.adj_head(adj)
        # adj = rearrange(adj, 'b n (h d) -> b h n d', h = h)
        # dots = dots + adj

        attn = self.attend(dots) 
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., has_global_token=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, adj_dim=83))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        # x, adj = data
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x 

class PolarRec_Encoder(nn.Module):
    def __init__(self, *,  
            input_dim, pe_dim, dim, depth, heads, mlp_dim,  
            dim_head = 16, dropout = 0., emb_dropout = 0., 
            output_dim = 1024, output_tokens = 3, 
            has_global_token=False, group_size = 16,
            ):

        '''
        dim_value_embedding = dim - pe_dim
        '''

        super().__init__()

        assert output_tokens>0

        assert dim>pe_dim

        self.input_dim=input_dim #input value dimension (e.g. =2 visibility map - real and imag. )
        self.pe_dim = pe_dim # positional-encoding dim.
        self.dim =dim #feature embedding dimension

        self.depth =depth
        self.heads =heads # number of multi-heads
        self.mlp_dim =mlp_dim
        self.output_dim = output_dim 
        self.output_tokens = output_tokens # number of output tokens

        self.global_token=None
        self.has_global_token= has_global_token # if use global token
        self.cen_dim = 0
        self.total_num = 1660
        self.mean_pool_size = 1660 // group_size

        if has_global_token:
            self.global_token = nn.Parameter(torch.randn(1, 1, dim))

        self.feat_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.dim - self.pe_dim - self.cen_dim),
        )

        self.before_mean = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, self.dim)
        )


        self.mean_pool = nn.AdaptiveAvgPool2d((self.mean_pool_size, self.dim))

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.output_token_heads = [ nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, self.output_dim)
                ) for _ in range(int(self.output_tokens)) ]
        self.output_token_heads = nn.ModuleList(self.output_token_heads)
        self.sorted_indices = None




    def adj_mat(self, pos):

        x = pos[:, :, 0]
        y = pos[:, :, 1]
        x = x.unsqueeze(2)
        y = y.unsqueeze(2)
        x_i_2 = torch.einsum('ijk,ijk->ijk',[x,x])
        res = x_i_2.expand(pos.shape[0], pos.shape[1], pos.shape[1])
        res = res + res.transpose(1,2)
        res = res - 2* torch.bmm(x, x.transpose(1,2))
        y_i_2 = torch.einsum('ijk,ijk->ijk',[y,y])
        y_2 = y_i_2.expand(pos.shape[0], pos.shape[1], pos.shape[1])
        res = res + y_2
        res = res + y_2.transpose(1,2)
        res = res - 2* torch.bmm(y, y.transpose(1,2))
        return F.normalize(res.sqrt(), p=2, dim=2) 



    def forward(self, tokens, pos):
        '''
        INPUTS
        tokens - B x N_token x Dim_token_feature(input_dim), input : [pose_embed, values]

        OUTPUTS:
        output_tokens -  output tokens (B x N_out_tokens x dim_out_tokens 
        '''
        # adj = self.adj_mat(pos)
        # cen_emb = self.central_embedding(adj)

        emb_val = self.feat_embedding(tokens[..., self.pe_dim :]) # B xN_token x self.dim- self.pe_dim


        # emb_token = torch.cat([tokens[..., :self.pe_dim], emb_val, cen_emb], dim=-1) # B xN_token x self.dim
        emb_token = torch.cat([tokens[..., :self.pe_dim], emb_val], dim=-1) # B xN_token x self.dim

        # if self.sorted_indices == None:
        #     batch_size = emb_token.shape[0]
        #     num_tokens = emb_token.shape[1]
            
        #     # We use only the first item's position in the batch for calculation
        #     pos_np = pos[0].cpu().detach().numpy()
            
        #     # Calculate pairwise distances
        #     distances = pairwise_distances(pos_np, pos_np)
        #     n = 20
        #     # Apply clustering
        #     clustering = AgglomerativeClustering(n_clusters=num_tokens//n, affinity='precomputed', linkage='average')
        #     labels = clustering.fit_predict(distances)
            
        #     # Now we sort our tokens and pos based on labels
        #     self.sorted_indices = torch.argsort(torch.tensor(labels))

        if self.sorted_indices == None:
            batch_size = emb_token.shape[0]
            num_tokens = emb_token.shape[1]
            
            # We use only the first item's position in the batch for calculation
            pos_np = pos[0].cpu().detach().numpy()
            
            x_coords = pos_np[:, 0]
            y_coords = pos_np[:, 1]

            angles = np.arctan2(y_coords, x_coords)
            
            self.sorted_indices = torch.argsort(torch.tensor(angles))  

            np.save("vispoints.npy", pos_np)
        

        emb_token = emb_token[:, self.sorted_indices]


        emb_token = self.before_mean(emb_token)
        emb_token = self.mean_pool(emb_token)
     

 
        B, N_token, _  = emb_token.shape

        if self.has_global_token:
            emb_token = torch.cat([self.global_token.repeat(B, 1,1), emb_token], dim=1)

        emb_token = self.emb_dropout(emb_token)
        transformed_token = self.transformer(emb_token) 

        #currently use the index reduction but there are other reduction 
        #TODO: use matrix multiplication as the reduction method
        transformed_token_reduced = transformed_token[:, :self.output_tokens, ...]

      
        out_tokens=[]
        for idx_token in range(self.output_tokens):
            out_tokens.append(self.output_token_heads[idx_token](
                transformed_token_reduced[:, idx_token,...].unsqueeze(1)))
        output_tokens = torch.cat(out_tokens, dim=1)


        return output_tokens


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class QuickFix(nn.Module):
    def __init__(self, dim, heads, fn):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.linear = nn.Linear(dim * heads, dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.linear(self.fn(x, **kwargs))



#ConvEncoder#
class LinearEncoder(nn.Module):
    def __init__(self, x_dim=28*28, hidden_dims=[512, 256], latent_dim=2, in_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(x_dim*in_channels, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc31 = nn.Linear(hidden_dims[1], latent_dim) # mu
        self.fc32 = nn.Linear(hidden_dims[1], latent_dim) # log_var       
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var

class ConvEncoder(nn.Module):
    def __init__(self, x_dim=28*28, hidden_dims=[32, 64], latent_dim=2, in_channels=1, activation=nn.ReLU):
        super().__init__()
        self.activation = activation
        modules = []
        '''modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                              kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(hidden_dims[0]),
                    activation()))
        modules.append(nn.Sequential(
                    nn.Conv2d(hidden_dims[0], out_channels=2*hidden_dims[0],
                              kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(2*hidden_dims[0]),
                    activation()))
        in_channels = 2*hidden_dims[0]'''
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(h_dim),
                    activation())
            )
            in_channels = h_dim
        #bottleneck_res = [28, 14, 7, 4, 2] + [1]*30 ## TODO only valid for 28^2 mnist digits
        bottleneck_res = [int(np.ceil(np.sqrt(x_dim) * 0.5**i)) for i in range(35)] # set res to decrease geometrically
        self.res_flattened = bottleneck_res[len(hidden_dims)]
        self.encoder = nn.Sequential(*modules)
        self.fc_mu =  nn.Linear(hidden_dims[-1]*(self.res_flattened**2), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*(self.res_flattened**2), latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        try:
            mu = self.fc_mu(x)
        except:
            import ipdb; ipdb.set_trace()
        log_var = self.fc_var(x)

        return mu, log_var
        





