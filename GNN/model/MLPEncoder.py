import torch
from torch import nn
from torch_geometric.nn import MLP
import numpy as np
import torch.nn.functional as F

class PyG_MLPEncoder(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_channels=[256, 512], k=2, dropout=0.0):
        super(PyG_MLPEncoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(MLP([in_feats, hidden_channels[0]]))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels[0]))

        for i in range(k - 1):
            self.convs.append(MLP([hidden_channels[i], hidden_channels[i+1]]))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels[i+1]))
        
        self.linear = torch.nn.Linear(hidden_channels[-1], out_feats)

        self.dropout_rate = dropout

    

    def forward(self, x, edge_index):
        # x, edge_index = data["feature"], data["edge_index"]

        # print(x.shape, edge_index.shape)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.linear(x)

        return x