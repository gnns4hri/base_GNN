
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from functools import partial

class RGCNLayer(nn.Module):
    def __init__(self, g, in_feat, out_feat, num_rels, bias=None, activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.g = g
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # weight in equation (3)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.in_feat), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat), requires_grad=True)
        # add bias
        if self.bias:
            self.bias1 = nn.Parameter(torch.Tensor(in_feat, 1), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(out_feat, 1), requires_grad=True)
            print(self.bias1.shape)
            print(self.bias2.shape)

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight2, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias1, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bias2, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs):
        self.g.ndata.update({'h': inputs})
        weight1 = self.weight1
        weight2 = self.weight2
        def message_func(edges):
            w1 = weight1[edges.data['rel_type'].squeeze()]
            w2 = weight2[edges.data['rel_type'].squeeze()]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w1).squeeze(1)
            if self.bias:
                msg = msg + self.bias1.squeeze(1)
            msg = torch.bmm(self.activation(msg).unsqueeze(1), w2).squeeze(1)
            msg = msg * edges.data['norm']
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias2.squeeze(1)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        self.g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
        return self.g.ndata['h']


class RGCN(nn.Module):
    def __init__(self, g, in_dim, h_dim, out_dim, num_rels, num_hidden_layers=1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers

        # create RGCN layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{}'.format(self.in_dim, self.h_dim))
        return RGCNLayer(self.g, self.in_dim, self.h_dim, self.num_rels, bias=True, activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        print('Building an HIDDEN  layer of {}x{}'.format(self.h_dim, self.h_dim))
        return RGCNLayer(self.g, self.h_dim, self.h_dim,  self.num_rels, bias=True, activation=F.relu)

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.h_dim, self.out_dim))
        return RGCNLayer(self.g, self.h_dim, self.out_dim, self.num_rels, bias=True, activation=F.relu)
        # return RGCNLayer(self.g, self.h_dim, self.out_dim, self.num_rels, bias=True, activation=partial(F.softmax, dim=1))

    def forward(self, features):
        h = features
        self.g.ndata['h'] = features
        for layer in self.layers:
            h = layer(h)
        ret = self.g.ndata.pop('h')
        # print ('RET: ({})'.format(ret.shape))
        return ret
