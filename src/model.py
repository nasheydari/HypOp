from torch.nn.parameter import Parameter
import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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
            #init.zeros_(self.weight)
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


class HyperGraphAttentionLayerSparse(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer=True, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer


        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[0]  # number of node

        pair = adj.nonzero().t()

        # get = lambda i: x_4att[adj[i].nonzero(),:]
        # x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])


        q1 = self.word_context.weight[0:].view(1, -1).repeat(x_4att.shape[0], 1).view(x_4att.shape[0], self.out_features)

        pair_h = torch.cat((q1, x_4att), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        # e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()
        e = adj * pair_e.repeat(N1,1).t()
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=0)

        edge = torch.matmul(attention_edge.t(), x)

        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        # get = lambda i: edge_4att[i][adj[i].nonzero().t()]
        # y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        # get = lambda i: x_4att[i][adj[i].nonzero().t()]
        # q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((x_4att[:, None, :].expand(-1, N1, -1), edge_4att[None, :, :].expand(N2, -1, -1)), dim=2)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze())
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        # e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()
        e = adj * pair_e
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention, dim=1)

        node = torch.matmul(attention_node, edge)

        if self.concat:
            # node = F.elu(node)
            node=torch.relu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class HGNN_ATT(nn.Module):
    def __init__(self, n, input_size, n_hid, output_size, params):
        super(HGNN_ATT, self).__init__()
        self.dropout = params["dropout"]
        self.embedding=nn.Embedding(n,input_size)
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2,
                                                   transfer=True, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2,
                                                   transfer=True, concat=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def forward(self, H):
        x = self.embedding.weight
        x = self.gat1(x, H)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, H)
        x = torch.sigmoid(x)
        return x