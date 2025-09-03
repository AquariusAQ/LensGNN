
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
import numpy as np
import torch
from tqdm import tqdm

from load_cora import get_raw_text_cora as get_raw_text
data, text = get_raw_text(use_text=True, root="../")


data.num_nodes = torch.tensor(data.num_nodes)
for key in data.keys():
    if not torch.is_tensor(data[key]):
        data[key] = torch.from_numpy(data[key])

subdata_list = []
for i in tqdm(range(data.num_nodes)):
# for i in tqdm(range(1)):
    data_subgraphs = k_hop_subgraph(node_idx=[i], num_hops=2, edge_index=data.edge_index)
    subset = data_subgraphs[0]
    subdata = data.subgraph(subset)
    print(subdata)
    subdata = Data(x=subdata.x, edge_index=subdata.edge_index)
    subdata_list.append(subdata)
    print(subdata)
    break

