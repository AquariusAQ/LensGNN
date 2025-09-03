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

        self.convs.append(GCNConv(in_feats, hidden_channels[0]))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels[0]))

        for i in range(k - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels[i+1]))
        
        self.linear = torch.nn.Linear(hidden_channels[-1], out_feats)
        # self.conv1d = nn.Conv1d(in_channels=hidden_channels[-1], out_channels=out_feats, kernel_size=1, bias=True)

        self.dropout_rate = dropout

    

    def forward(self, x, edge_index=None):
        # x, edge_index = data["feature"], data["edge_index"]

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            # x = F.normalize(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)


        # out = []
        # import math
        # from tqdm import tqdm
        # batch_size = 1024
        # batch = math.ceil(x.shape[0] / batch_size)
        # # for i in tqdm(range(batch)):
        # for i in range(batch):
        #     out.append(self.linear(x[i*batch_size:min((i+1)*batch_size, x.shape[0])]))
        # out = torch.cat(out,dim=0)
        # print(out.shape)
        # exit(0)

        out = self.linear(x)

        # 调整输入数据形状以适应一维卷积层
        # x_reshaped = x.unsqueeze(2)  # shape变为(batch_size, 32, 1)
        # # 应用一维卷积层
        # conv_output = self.conv1d(x_reshaped)
        # # 调整输出数据形状以匹配全连接层的输出
        # out = conv_output.squeeze(2)  # shape恢复为(batch_size, 40960)

        return out