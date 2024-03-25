import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np

def posenc(x, L_embed=4):

  rets = [x]
  for i in range(0, L_embed):
    for fn in [torch.sin, torch.cos]:
      rets.append(fn(2.*3.14159265*(i+1) * x))
  return torch.cat(rets, dim=-1)
  
def calcB(m=1024, d=2, sigma=1.0):
    B = torch.randn(m, d)*sigma
    return B.cuda()
    
def fourierfeat_enc(x, B):

    feat = torch.cat([#torch.sum(x**2, -1, keepdims=True), ## new
                      x, ## new
                      torch.cos(2*3.14159265*(x @ B.T)),
                      torch.sin(2*3.14159265*(x @ B.T))], -1)
    return feat

class PE_Module(torch.nn.Module):
    def __init__(self, type, embed_L):
        super(PE_Module, self).__init__()

        self.embed_L= embed_L
        self.type=type

    def forward(self, x):
        if self.type == 'posenc':
            return posenc(x, L_embed=self.embed_L)

        elif self.type== 'fourier':
            return fourierfeat_enc(x, B=self.embed_L)

class PosEncodedMLP(torch.nn.Module):
    def __init__(self, 
            input_size=2, output_size=2, 
            hidden_dims=[256, 256], L_embed=5, 
            embed_type='nerf', activation=nn.ReLU, sigma=0.1,
            ):

        super(PosEncodedMLP, self).__init__()
        self.embed_type = embed_type
        self.L_embed = L_embed
        if self.L_embed > 0 and self.embed_type == 'nerf':
            self.input_size = L_embed*2*input_size+input_size
        elif self.L_embed > 0 and self.embed_type == 'fourier':
            self.B = calcB(m=L_embed, d=2, sigma=sigma)
            self.input_size = L_embed*2+3
        else:
            self.input_size = input_size

        #import ipdb; ipdb.set_trace()

        modules = []
        dim_prev = self.input_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(dim_prev, h_dim),
                    activation()))
            dim_prev = h_dim           
        modules.append(nn.Sequential(nn.Linear(hidden_dims[-1], output_size),
                                     ))#nn.Sigmoid()))
        self.mlp = nn.Sequential(*modules)        
    
    def _step(self, x):
        
        if self.L_embed > 0 and self.embed_type == 'nerf': 
            x = posenc(x, self.L_embed)      
        elif self.L_embed > 0 and self.embed_type == 'fourier':
            x = fourierfeat_enc(x, self.B)

        x = self.mlp(x)      

        return x     
      
    def forward(self, x):
        x = self._step(x)
        return x
        


class PosEncodedMLP_FiLM(pl.LightningModule):

    def __init__(self, context_dim=64, input_size=2, output_size=2,
                 hidden_dims=[256, 256], L_embed=10, embed_type='nerf',
                 activation=nn.ReLU, sigma=5.0,
                 context_type='VAE'):
        '''
        context_type = 'VAE'(default) | 'Transformer'
        '''
        super().__init__()

        self.context_type = context_type
        if context_dim > 0:
            layer = FiLMLinear
        else:
            layer = nn.Linear # will break if context_dim is an input
            
        self.context_dim = context_dim
        
        self.embed_type = embed_type
        self.L_embed = L_embed 


        if self.L_embed > 0 and self.embed_type == 'nerf':
            self.input_size = L_embed*2*input_size+input_size
        elif self.L_embed > 0 and self.embed_type == 'fourier':
            self.B = nn.Parameter(calcB(m=L_embed, d=2, sigma=sigma), requires_grad=False)
            # self.input_size = L_embed*2+3
            self.input_size = L_embed*2+2 # change from +3 to +2 due to change in the fourierfeat_enc() function  
        else:
            self.input_size = input_size

        #positional embedding function#
        if self.L_embed > 0 and self.embed_type == 'nerf': 
            # self.embed_fun = lambda x_in: posenc(x_in, self.L_embed)      
            self.embed_func= PE_Module(type='posenc', embed_L=self.L_embed)
            
        elif self.L_embed > 0 and self.embed_type == 'fourier':
            # self.embed_fun = lambda x_in: fourierfeat_enc(x_in, self.B)
            self.embed_fun = PE_Module(type='fourier', embed_L= self.B)
        
        self.layers = []
        self.activations = []
        dim_prev = self.input_size
        for h_dim in hidden_dims:
            self.layers.append(layer(dim_prev, h_dim, context_dim=self.context_dim))
            self.activations.append(activation())
            dim_prev = h_dim

        # self.layer1 = layer(self.input_size, hidden_dims[0], context_dim=self.context_dim)
        # self.act1 = activation()
        # self.layer2 = layer(hidden_dims[0], hidden_dims[1], context_dim=self.context_dim)
        # self.act2 = activation()

        self.layers= nn.ModuleList(self.layers)
        self.activations= nn.ModuleList(self.activations)
        self.final_layer = layer(hidden_dims[-1], output_size, context_dim=self.context_dim)
        ##self.final_activation = nn.Sigmoid() ## TODO removed this for unconstrained output
    
    def set_B(self, B):
        self.B = B
    
    def forward(self, x_in, context):
        '''
        context - 
        B x 1 x ndim for VAE, 
        B x L x ndim for Transfomer (assuming L layers in MLP)
        '''

        # if self.L_embed > 0 and self.embed_type == 'nerf': 
        #     x_embed = posenc(x_in, self.L_embed)      
        # elif self.L_embed > 0 and self.embed_type == 'fourier':
        #     x_embed = fourierfeat_enc(x_in, self.B)

        x_embed = self.embed_fun(x_in) # B x N x 2 -> B x N x dim_PE_dim
        
     
        #for l, a in zip(self.layers, self.activations):
        #    print(x.shape, x.device, context.shape, context.device); input()
        #    x = l(x, context)
        #    x = a(x)

        # if self.context_type=='VAE':
        #     con1 = context 
        #     con2 = context
        #     con3 = context
        # elif self.context_type=='Transformer':
        #     con1 = context[:, 0, :].unsqueeze(1)
        #     con2 = context[:, 1, :].unsqueeze(1)
        #     con3 = context[:, 2, :].unsqueeze(1)
        # x = self.layer1(x_embed, con1)
        # x = self.act1(x)
        # x = self.layer2(x, con2)
        # x = self.act2(x)
        # x = self.final_layer(x, con3)
        #x = self.final_activation(x)              

        x_tmp = x_embed
        for ilayer, layer in enumerate(self.layers):
            x = layer( x_tmp, context if self.context_type=='VAE' else context[:,ilayer,:].unsqueeze(1) )
            x = self.activations[ilayer](x)
            x_tmp = x

        x= self.final_layer(x_tmp, context if self.context_type=='VAE' else context[:,-1,:].unsqueeze(1) )

        return x     

class FiLMLinear(pl.LightningModule):
    def __init__(self, in_dim, out_dim, context_dim=64, residual=False):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()
        self.film1 = nn.Linear(context_dim, out_dim)
        self.film2 = nn.Linear(context_dim, out_dim)        
        self.residual = residual

    def forward(self, x, shape_context):
        if self.residual:
            out = self.linear(x)
            resid = self.activation1(out)
        
            gamma = self.film1(shape_context)
            beta = self.film2(shape_context)

            out = gamma * out + beta

            out = self.activation2(out)
            out = out + resid
        else:
            out = self.linear(x)
            gamma = self.film1(shape_context)
            beta = self.film2(shape_context)
            out = gamma * out + beta
            out = self.activation1(out)
        return out


class Linear(pl.LightningModule):
    ''' dummy wrapper around linear to support (ignoring) shape context param'''
    def __init__(self, in_dim, out_dim, context_dim=64, residual=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, shape_context=None): #ignore shape context
        out = self.linear(x)
        return out

import torch
from torch import nn

class NeRF_Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(NeRF_Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class NeRF_Fourier(pl.LightningModule):
    def __init__(self,
                 context_dim=64,
                 input_size=2,
                 output_size=5,
                 D=8, W=256,
                 L_embed=10,
                 skips=[4],
                 hidden_dims = None, # dummy
                 embed_type = 'nerf',
                 activation = nn.ReLU,
                 sigma = 2.5,
                 context_type='VAE'):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        """
        super(NeRF_Fourier, self).__init__()

        self.context_type = context_type
        if context_dim > 0:
            Layer = FiLMLinear
        else:
            Layer = Linear

        self.context_dim = context_dim

        self.embed_type = embed_type
        self.L_embed = L_embed

        self.D = D
        self.W = W

        self.skips = skips

        if embed_type == 'nerf':
            self.embedding_xyz = NeRF_Embedding(input_size, L_embed, logscale=True)  # 10 is the default number
            self.in_channels_xyz = input_size * (
                        len(self.embedding_xyz.funcs) * self.embedding_xyz.N_freqs + 1)  # in_channels_xyz
        else:
            self.B = calcB(m=L_embed, d=input_size, sigma=sigma)
            self.in_channels_xyz = L_embed*2  + input_size #+ 1 #

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = Layer(self.in_channels_xyz, W, context_dim=self.context_dim)
            elif i in skips:
                layer = Layer(W+self.in_channels_xyz, W, context_dim=self.context_dim)
            else:
                layer = Layer(W, W, context_dim=self.context_dim)
            layer = _Sequential(layer, activation(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = Layer(W, W, context_dim=self.context_dim)

        # output layers (real and imag)
        # or if using phase loss, out_dim may be 5
        self.fourier = Layer(W, output_size, context_dim=self.context_dim)

    def set_B(self, B):
        self.B = B

    def forward(self, x, context=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if self.embed_type == 'nerf':
            embedded_x = self.embedding_xyz(x)
        else:
            embedded_x = fourierfeat_enc(x, self.B)
        input_xyz = embedded_x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_, context)

        fourier = self.fourier(xyz_, context)

        return fourier

class _Sequential(nn.Sequential):
    def forward(self, input, shape_context=None):
        for module in self._modules.values():
            if type(module) == FiLMLinear or type(module) == _Sequential:
                input = module(input, shape_context=shape_context)
            else:
                input = module(input)
        return input

class NeRF_Fourier_Two_Heads(nn.Module):
    def __init__(self,
                 input_size=2,
                 output_size=5,
                 D=8, W=256,
                 L_embed=10,
                 skips=[4],
                 embed_type = 'nerf',
                 activation = nn.ReLU,
                 sigma = 2.5,
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        """
        super(NeRF_Fourier_Two_Heads, self).__init__()
        self.D = D
        self.W = W

        self.skips = skips

        self.embed_type = embed_type

        if embed_type == 'nerf':
            self.embedding_xyz = NeRF_Embedding(input_size, L_embed, logscale=True)  # 10 is the default number
            self.in_channels_xyz = input_size * (
                        len(self.embedding_xyz.funcs) * self.embedding_xyz.N_freqs + 1)  # in_channels_xyz
        else:
            self.B = calcB(m=L_embed, d=input_size, sigma=sigma)
            self.in_channels_xyz = L_embed*2 + input_size + 1

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        #self.xyz_encoding_final = nn.Linear(W, W)

        # output layers (real and imag)
        # or if using phase loss, out_dim may be 5

        self.ampl = nn.Sequential(
                        nn.Linear(W, W),
                        activation(True),
                        nn.Linear(W, 1))
        self.phase = nn.Sequential(
                        nn.Linear(W, W),
                        activation(True),
                        nn.Linear(W, output_size-1))

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if self.embed_type == 'nerf':
            embedded_x = self.embedding_xyz(x)
        else:
            embedded_x = fourierfeat_enc(x, self.B)
        input_xyz = embedded_x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        amp = self.ampl(xyz_)
        phase = self.phase(xyz_)

        return torch.cat([amp, phase], -1)
