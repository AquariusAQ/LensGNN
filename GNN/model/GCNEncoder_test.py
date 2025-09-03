import torch
from torch import nn
from torch_geometric.nn import GCNConv
import numpy as np
import torch.nn.functional as F

class PyG_GCNEncoder(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_channels=[256, 512], k=2, dropout=0.0):
        super(PyG_GCNEncoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_feats, hidden_channels[0], normalize=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels[0]))
        self.lns.append(nn.Linear(in_feats, hidden_channels[0]))

        for i in range(k - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1], normalize=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels[i+1]))
            self.lns.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
        
        self.linear = torch.nn.Linear(hidden_channels[-1], out_feats)

        self.dropout_rate = dropout

    

    def forward(self, x, edge_index):
        # x, edge_index = data["feature"], data["edge_index"]

        # print(x.shape, edge_index.shape)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.dropout(x, p=0.15, training=self.training)

        x_local = 0
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) + self.lns[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x_local = x_local + x

        # x = self.linear(x_local)

        return x_local