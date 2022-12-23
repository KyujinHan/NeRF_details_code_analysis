# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 04:14:44 2022

@author: KyujinHan
- D_NeRF model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# First, positional emebedding
# Each coordinates applying [sin, cos] function.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn() # function below
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims'] # input dimension
        out_dim = 0
        if self.kwargs['include_input']: # If you want the original (x,y,z) coordinates
            embed_fns.append(lambda x : x)
            out_dim += d # Usually, 3
        
        # max_freq_log2: L-1
        # N_freqs: L
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        # If you make positional embedding method using paper's method
        # 1, 2, 4, 8, 16, ...
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            
        # It just linear method
        # Spacing is not equal
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']: # self.kwargs['periodic_fns'] = [torch.sin, torch.cos]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq)) # make function
                out_dim += d # Usually 63 or 63
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim # Usually 60 or 63
    
    # Make applying positional embedding function
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# get embedding function
def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Second, make model network

# First, NeRF original.
# It is Canonical network
class Canonical_NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, output_color_ch=3, zero_canonical=True):
        super(Canonical_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch # Almost 63 or 60
        self.input_ch_views = input_ch_views # Almost 27 or 24
        self.skips = skips # Where input again (x,y,z)
        self.use_viewdirs = use_viewdirs # Boolean

        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] +
        #     [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # Make linear function
        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            # If next linear input again (x,y,z) n-dimension.
            if i in self.skips:
                in_channels += input_ch # 256 + 63 or 60

            layers += [layer(in_channels, W)]
        
        self.pts_linears = nn.ModuleList(layers) # until, before density output layer.

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)]) # input view directions and next linear is 128
        
        # Almost we using.
        if use_viewdirs: 
            self.feature_linear = nn.Linear(W, W) # 256 -> 256 (density output stage)
            self.alpha_linear = nn.Linear(W, 1) # Output: Density
            self.rgb_linear = nn.Linear(W//2, output_color_ch) # Output: RGB
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    # Canonical space not using t variables.
    # x shape: [N, Coordinates_positional_embedding + View_directions_positional_embedding]
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1) # So, must split
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips: # input again (x,y,z)
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h) # Density
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h) # RGB
            outputs = torch.cat([rgb, alpha], -1) # concat, make 4-dimension
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
        
        
#################################################
# Second, make Deformation network
# And D-NeRF class function
class D_NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(D_NeRF, self).__init__()
        # Also same Canonocial network
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views 
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn # positional embedding function
        self.zero_canonical = zero_canonical # Boolean, If you want not using t=0, false. // Almost True
        
        # Canonical network
        self._ca_nerf = Canonical_NeRF(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views,
                                 input_ch_time=input_ch_time, output_ch=output_ch, skips=skips,
                                 use_viewdirs=use_viewdirs, memory=memory, embed_fn=embed_fn, output_color_ch=3)
        
        #  # Create deformation network
        self._deformation_layers, self._deformation_out_layer = self.create_deformation_net()
    
    # The structure almost same the Canonical network
    def create_deformation_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips: 
                in_channels += self.input_ch # 256 + 63 or 60
            
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)
    
    # implementation deformation network
    def deformation_network_imp(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1) # Dimension ==> new_pts: 60 or 63 // t: 10 or 11
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts):
        # X will be concat, the X positional embedding and View directions positional embedding
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # N: The number of t is sampling number.
        # 10 or 11: positional embedding
        t = ts[0] # t's shape [N,10 or 11] 

        # Must same! Because, D-NeRF sampling cooridnate in one image.
        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time" 
        
        cur_time = t[0, 0]
        if cur_time == 0. and self.zero_canonical: # If t = 0
            dx = torch.zeros_like(input_pts[:, :3]) # dx = 0
            
        else:
            dx = self.deformation_network_imp(input_pts, t, self._deformation_layers, self._deformation_out_layer) # Deformation network
            input_pts_orig = input_pts[:, :3] # (x,y,z)
            input_pts = self.embed_fn(input_pts_orig + dx) # (x+dx, y+dx, z+dx) applying position embedding
            
        # out = [RGB. density] 4-dimension
        out = self._ca_nerf(torch.cat([input_pts, input_views], dim=-1)) # And input above this. 
        return out, dx
    

###########################################
# Third, Make all NeRF class
class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("NeRF type selected: %s" % type)

        if type == "original":
            model = Canonical_NeRF(*args, **kwargs)
        elif type == "D_NeRF": # almost we using this.
            model = D_NeRF(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model