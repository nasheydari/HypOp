from torch.nn.parameter import Parameter
import torch
import math
import torch.nn as nn
import torch.nn.init as init

class single_node(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(single_node, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
    
class single_node_xavier(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(single_node_xavier, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    
class GCN_decentral_sim():
    def __init__(self, X, layer, dct, idx, G):
        self.x = X
        self.layer = layer
        self.dct = dct
        self.idx = idx
        self.G = G
    
    def _get_info(self):
        return self.x
            
    def __fetch_info(self, aggregate):
        input_x = []
        for node in self.dct:
            if node == self.idx:
                input_x.append(aggregate[self.idx].clone())
            else:
                input_x.append(aggregate[node].clone().detach())
        self.f = torch.cat(input_x)
            
    def __call__(self, aggregate):
        self.__fetch_info(aggregate)
        #print(self.layer, self.f.shape)
        self.f = self.G @ self.f
        self.f = self.layer(self.f)
        #print(self.f.shape)
        self.f = torch.relu(self.f)
        return self.f
    
class GraphSage_decental_sim():
    def __init__(self, X, layer, dct, idx, spread=0.8):
        #print(X.shape)
        self.x = X
        self.layer = layer
        self.dct = dct
        self.idx = idx
        self.spread = spread
        
    def _get_info(self):
        return self.x
            
    def __fetch_info(self, aggregate):
        input_x = []
        for node in self.dct:
            if node == self.idx:
                input_x.append(aggregate[self.idx].clone())
            else:
                input_x.append(aggregate[node].clone().detach())
        self.f = torch.cat(input_x, axis=0)
        self.f = torch.mean(self.f, axis=0, keepdim=True)
            
    def __call__(self, aggregate):
        self.__fetch_info(aggregate)
        
        #print(self.layer, self.f.shape, self.x.shape)
        self.f = torch.cat([self.x, self.f*self.spread], axis=1)
        #print(self.f.shape)
        self.f = self.layer(self.f)
        #print(self.f.shape)
        self.f = torch.relu(self.f)
        return self.f
    