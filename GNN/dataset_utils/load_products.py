from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os

def get_raw_text_products(use_text=False, seed=0, root="./"):

    # dataset = PygNodePropPredDataset(
    #     name='ogbn-arxiv', transform=T.ToSparseTensor())
    dataset = PygNodePropPredDataset(
        name='ogbn-products', root=os.path.join(root, "datasets/ogbn_products"))
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

    nodeidx2asin = pd.read_csv(
        os.path.join(root, 'datasets/ogbn_products/ogbn_products/mapping/nodeidx2asin.csv.gz'), compression='gzip')

    raw_text_trn = pd.read_json(os.path.join(root, 'datasets/ogbn_products/ogbn_products_orig/trn.json.gz'), lines=True, compression='gzip')
    raw_text_tst = pd.read_json(os.path.join(root, 'datasets/ogbn_products/ogbn_products_orig/tst.json.gz'), lines=True, compression='gzip')
    raw_text = pd.concat((raw_text_tst[["uid","title","content"]], raw_text_trn[["uid","title","content"]])).rename(columns={'uid': 'asin'})
    
    df = pd.merge(nodeidx2asin, raw_text, on='asin')
    # print(df)
    text = []
    for ti, ab in zip(df['title'], df['content']):
        t = 'Title: ' + ti + 'Content: ' + ab
        text.append(t)
    data.y = data.y.reshape(-1)
    return data, text