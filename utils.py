import os
import json
import torch

def read_gnn_encoder(root:str, config, device):
    with_gnn_feature = root.replace("-graph", "").split("-")[1:-1]

    with open(os.path.join(root, "config.json")) as f:
        gnn_config = json.load(f)
        f.close()
    with open(os.path.join(root, "history.json")) as f:
        history = json.load(f)
        f.close()
    
    encoder_list = []
    
    for gnn_name in with_gnn_feature:
        if gnn_name == "gcn":
            from GNN.model.GCNEncoder import PyG_GCNEncoder as model
        elif gnn_name == "gat":
            from GNN.model.GATEncoder import PyG_GATEncoder as model
        elif gnn_name == "gin":
            from GNN.model.GINEncoder import PyG_GINEncoder as model
        elif gnn_name == "sage":
            from GNN.model.GraphSAGEEncoder import PyG_SAGEEncoder as model
        elif gnn_name == "mlp":
            from GNN.model.MLPEncoder import PyG_MLPEncoder as model
        else:
            print(f"Cannot find model {gnn_name}")
            exit(0)
        encoder = model(in_feats=gnn_config["text_feature_size"], \
                               hidden_channels=gnn_config["hidden_channels"], \
                               out_feats=gnn_config["gnn_output_feature_size"], \
                                k=gnn_config["layer"], \
                                dropout=gnn_config["dropout"])
        encoder = encoder.to(device)
        print(f"Loading {gnn_name} encoder")
        encoder.load_state_dict(torch.load(os.path.join(root, f"{gnn_name}.pt")))
        encoder_list.append(encoder)
    return (encoder_list, gnn_config, history)

import hashlib

def hash_list(lst):
    tuple_repr = tuple(lst)
    hash_object = hashlib.sha256()
    hash_object.update(str(tuple_repr).encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return hex_dig

# def hash_list(l):
#     # return sum(hash(item) for item in l)
#     return hash(tuple(l))

# vec_data_path = "./vector_dataset/vector_dataset.pt"

import requests
import json
import tqdm

def save_to_vector_dataset(datas, vec_data_path):
    if os.path.exists(vec_data_path):
        data_old = torch.load(vec_data_path)
        data_old.update(datas)
    else:
        data_old = datas
    torch.save(data_old, vec_data_path)



import math
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from GNN.dataset_utils.dataset import get_dataset, build_subgraph_new
from tqdm import tqdm
def encoding(data, encoder, encode_range, config, gnn_config, device, device_subgraph):
    print("device", device)
    encoder = encoder.to(device)
    encoder.eval()

    feature_list = []
    
    batch_size = gnn_config["batch_size"]

    for batch_i in tqdm(range(math.ceil((encode_range[1]-encode_range[0]) / batch_size))):
    # for batch_i in range(math.ceil(data.train_mask.shape[0] / batch_size)):
        # print(torch.cuda.memory_allocated(4))
        if (batch_i+1)%1000 == 0:
            torch.cuda.empty_cache()
        sublist = build_subgraph_new(data, [i+encode_range[0] for i in range(batch_i*batch_size, min(encode_range[1]-encode_range[0],(batch_i+1)*batch_size))], gnn_config["layer"], device=device_subgraph)
        
        dataset_loader = DataLoader(sublist, batch_size=batch_size, shuffle=False)
        for _, batch in enumerate(dataset_loader):
            batch_x = batch.x.cpu().to(device)  # 
            batch_edge_index = batch.edge_index.cpu().to(device) # 
            batch_batch = batch.batch.cpu().to(device)
            hidden = encoder(batch_x, batch_edge_index)
            # print(hidden[0]) #
            feature = scatter(hidden, batch_batch, dim=0, reduce='mean')
            feature_list.append(feature.cpu())
    
    return torch.cat(feature_list, dim=0)

def encoding_graph(data, encoder, config, gnn_config, device, device_subgraph):
    print("device", device)
    encoder = encoder.to(device)
    encoder.eval()

    feature_list = []
    
    hidden = encoder(data.x, data.edge_index).cpu()
    
    return hidden