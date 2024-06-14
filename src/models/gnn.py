import torch
import dgl.nn.pytorch.conv as dglnn
from torch import nn

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.mlp(h, False)
        return h

    def getHiddenState(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='pool', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.SAGEConv(h_feats, h_feats, agg, activation=self.act))
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h
    
    def getHiddenState(self, graph):
        h = graph.ndata['feature']
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        return h

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h
    
class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,
                                 aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        print(h_final)
        return h_final