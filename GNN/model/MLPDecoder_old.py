import torch
from torch import nn
from torch_geometric.utils import scatter
import numpy as np
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_channels=[256], k=2, dropout=0.0):
        super(MLPDecoder, self).__init__()

        self.input_layer = torch.nn.Linear(in_feats, hidden_channels[0], bias=True)
        self.hidden_layers = torch.nn.Sequential()
        for i in range(2, k):
            self.hidden_layers.add_module(f"hidden {i-2}", torch.nn.Linear(hidden_channels[i-2], hidden_channels[i-1], bias=True))
            # self.hidden_layers.add_module(f"relu {i-2}", torch.nn.LeakyReLU())
        self.dropout = torch.nn.Dropout(p=dropout)
        self.output_layer = torch.nn.Linear(hidden_channels[-1], out_feats, bias=True)

    def forward(self, input):
        # print(input.shape)
        feature = self.input_layer(input)
        feature = self.dropout(feature)
        output = self.output_layer(feature)
        # output = torch.nn.functional.softmax(output)

        return output
