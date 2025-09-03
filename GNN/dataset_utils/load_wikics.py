from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os
import os.path
import numpy as np
import json
import itertools
import torch
from torch_geometric.data.data import Data

DATA_PATH = os.path.join('datasets', "wikics", 'data.json')

def load_data(filename=DATA_PATH, directed=False, root="./"):
    raw = json.load(open(os.path.join(root, filename)))
    features = torch.FloatTensor(np.array(raw['features']))
    labels = torch.LongTensor(np.array(raw['labels']))
    if hasattr(torch, 'BoolTensor'):
        train_masks = torch.BoolTensor(raw['train_masks'][0])
        val_masks = torch.BoolTensor(raw['val_masks'][0])
        stopping_masks = torch.BoolTensor(raw['stopping_masks'][0])
        test_mask = torch.BoolTensor(raw['test_mask'])
    else:
        train_masks = torch.ByteTensor(raw['train_masks'][0])
        val_masks = torch.ByteTensor(raw['val_masks'][0])
        stopping_masks = torch.ByteTensor(raw['stopping_masks'][0])
        test_mask = torch.ByteTensor(raw['test_mask'])

    if directed:
        edges = [[(i, j) for j in js] for i, js in enumerate(raw['links'])]
        edges = list(itertools.chain(*edges))
    else:
        edges = [[(i, j) for j in js] + [(j, i) for j in js]
                 for i, js in enumerate(raw['links'])]
        edges = list(set(itertools.chain(*edges)))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=features, edge_index=edge_index, y=labels)

    data.train_mask = train_masks
    data.val_mask = val_masks
    data.stopping_mask = stopping_masks
    data.test_mask = test_mask

    return data

def get_raw_text_wikics(use_text=False, seed=0, root="./"):

    data = load_data(root=root)

    data.train_id = data.train_mask.nonzero().squeeze()
    data.val_id = data.val_mask.nonzero().squeeze()
    data.test_id = data.test_mask.nonzero().squeeze()


    # print(data)
    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    metadata = json.load(open(os.path.join(root, 'datasets', "wikics", 'metadata.json')))
    
    # print(df)
    text = []
    for node in metadata["nodes"]:
        t = 'Title: ' + node["title"] + '\n' + 'Abstract: ' + " ".join(node["tokens"])
        text.append(t)

    data.y = data.y.reshape(-1)
    return data, text