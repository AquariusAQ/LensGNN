from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os

def get_raw_text_arxiv(use_text=False, seed=0, root="./"):

    # dataset = PygNodePropPredDataset(
    #     name='ogbn-arxiv', transform=T.ToSparseTensor())
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', root=os.path.join(root, "datasets/ogbn_arxiv"))
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(idx_splits['train'].shape, idx_splits['valid'].shape, idx_splits['test'].shape)
    data.train_id = idx_splits['train']
    data.val_id = idx_splits['valid']
    data.test_id = idx_splits['test']


    # print(data)
    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        os.path.join(root, 'datasets/ogbn_arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz'), compression='gzip')

    raw_text = pd.read_csv(os.path.join(root, 'datasets/ogbn_arxiv/ogbn_arxiv_orig/titleabs.tsv'),
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    
    # raw_text['paper id'] = raw_text['paper id'].apply(lambda x: int(x)) 
    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].apply(lambda x: str(x)) 
    # print(nodeidx2paperid)
    # print(raw_text)
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    # print(df)
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    data.y = data.y.reshape(-1)
    return data, text