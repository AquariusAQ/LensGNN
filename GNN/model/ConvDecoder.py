import torch
from torch import nn
from torch_geometric.utils import scatter
import numpy as np
import torch.nn.functional as F

class ConvDecoder(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_channels=[256], k=2, dropout=0.0):
        super(ConvDecoder, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, bias=True)
        # self.linear = torch.nn.Linear(in_feats, out_feats, bias=True)

    def forward(self, input):
        
        input_reshaped = input.unsqueeze(2)  # shape变为(batch_size, 32, 1)
        conv_output = self.conv1d(input_reshaped)
        output = conv_output.squeeze(2)

        return output
