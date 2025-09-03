import torch
from torch import nn
from torch_geometric.utils import scatter
import numpy as np
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_channels=[256], k=2, dropout=0.0):
        super(MLPDecoder, self).__init__()

        # self.input_layer = torch.nn.Linear(in_feats, hidden_channels[0], bias=True)
        # self.hidden_layers = torch.nn.Sequential()
        # for i in range(2, k):
        #     self.hidden_layers.add_module(f"hidden {i-2}", torch.nn.Linear(hidden_channels[i-2], hidden_channels[i-1], bias=True))
        #     # self.hidden_layers.add_module(f"relu {i-2}", torch.nn.LeakyReLU())
        # self.dropout = torch.nn.Dropout(p=dropout)
        # self.output_layer = torch.nn.Linear(hidden_channels[-1], out_feats, bias=True)

        self.linear = torch.nn.Linear(in_feats, out_feats, bias=True)

    def forward(self, input):
        # print(input.shape)
        # feature = self.input_layer(input)
        # feature = self.dropout(feature)


        output = self.linear(input)
        # out = []
        # x = input
        # import math
        # from tqdm import tqdm
        # batch_size = 1024
        # batch = math.ceil(x.shape[0] / batch_size)
        # # for i in tqdm(range(batch)):
        # for i in range(batch):
        #     out.append(self.linear(x[i*batch_size:min((i+1)*batch_size, x.shape[0])]))
        # out = torch.cat(out,dim=0)

        # output = torch.nn.functional.softmax(output)

        return output
