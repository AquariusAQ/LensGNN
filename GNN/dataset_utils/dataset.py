
import os
import tqdm

# seed = 240229

def get_dataset(dataset, lm="token", adj=False, enable_tokenize_cache=True, seed=0, root="./", device="cuda"):
    # device = "cpu"
    import torch
    if dataset == "ogbn_arxiv":
        from .load_arxiv import get_raw_text_arxiv as get_raw_text
        out_feats = 40
    elif dataset == "pubmed":
        from .load_pubmed import get_raw_text_pubmed as get_raw_text
        out_feats = 3
    elif dataset == "cora":
        from .load_cora import get_raw_text_cora as get_raw_text
        out_feats = 7
    elif dataset == "ogbn_products":
        from .load_products import get_raw_text_products as get_raw_text
        out_feats = 47
    elif dataset == "wikics":
        from .load_wikics import get_raw_text_wikics as get_raw_text
        out_feats = 10
    else:
        print(f"ERROR: Cannot find dataset {dataset}")
        return
    
    data, text = get_raw_text(use_text=False, seed=seed, root=root)
    # print(data)
    edge = data.edge_index
    print("Reversing...")
    edge_reverse = torch.stack([edge[1], edge[0]], dim=0)
    print("Catting...")
    data.edge_index = torch.cat([edge, edge_reverse], dim=1)
    print("Uniquing...")
    data.edge_index = torch.unique(data.edge_index, dim=1)
    print("Okay")
    # print(data)
    # exit(0)
    data.num_nodes = torch.tensor(data.num_nodes)
    # print(data)

    # adj
    # if adj:
    #     edge_num = data["edge_index"].shape[1]
    #     edge_index = torch.sparse_coo_tensor(indices=data["edge_index"], values=torch.ones(edge_num))
    #     adj = edge_index.to_dense().to(device)
    #     data["adj"] = adj
        # print(adj.shape)

    # feature
    if lm == "token":
        try:
            from model.lm.bert_token import tokenize
        except:
            from ..model.lm.bert_token import tokenize
        # if dataset == "ogbn_arxiv":
        #     feature = tokenize(text, enable_cache=enable_tokenize_cache, cache_root="datasets/ogbn_arxiv/ogbn_arxiv_orig_tokenize/")
        # elif dataset == "pubmed":
        #     feature = tokenize(text, enable_cache=enable_tokenize_cache, cache_root="datasets/pubmed/pubmed_orig_tokenize/")
        # elif dataset == "cora":
        #     feature = tokenize(text, enable_cache=enable_tokenize_cache, cache_root="datasets/cora/cora_orig_tokenize/")
        # elif dataset == "ogbn_products":
        #     feature = tokenize(text, enable_cache=enable_tokenize_cache, cache_root="datasets/ogbn_products/ogbn_products_orig_tokenize/")
        
        feature = tokenize(text, enable_cache=enable_tokenize_cache, cache_root=os.path.join(root, f"datasets/{dataset}/{dataset}_orig_tokenize/"))
        data.x = feature
    elif lm == "raw_text":
        data.x = text
    elif lm == "embed":
        try:
            from model.lm.bert_embedding import embed
        except:
            from ..model.lm.bert_embedding import embed
        if dataset == "ogbn_arxiv":
            # feature = embed(text, enable_cache=enable_tokenize_cache, cache_root="datasets/ogbn_arxiv/ogbn_arxiv_orig_tokenize/", device=device)
            out_feats = 40
        elif dataset == "pubmed":
            # feature = embed(text, enable_cache=enable_tokenize_cache, cache_root="datasets/pubmed/pubmed_orig_tokenize/", device=device)
            out_feats = 3
        elif dataset == "cora":
            # feature = embed(text, enable_cache=enable_tokenize_cache, cache_root="datasets/cora/cora_orig_tokenize/", device=device)
            out_feats = 7
        elif dataset == "ogbn_products":
            # feature = embed(text, enable_cache=enable_tokenize_cache, cache_root="datasets/ogbn_products/ogbn_products_orig_tokenize/", device=device)
            out_feats = 47
        elif dataset == "wikics":
            out_feats = 10
        feature = embed(text, enable_cache=enable_tokenize_cache, cache_root=os.path.join(root, f"datasets/{dataset}/{dataset}_orig_tokenize/"), device=device)
        import numpy as np
        data.x = torch.from_numpy(np.array(feature)).float()
        # data.x = torch.nn.functional.normalize(torch.from_numpy(feature), p=2, dim=-1).numpy()
        # torch.functional.norm
    else:
        print(f"ERROR: Cannot find lm {lm}")
        return 
    
    for key in data.keys():
        if not torch.is_tensor(data[key]):
            data[key] = torch.from_numpy(data[key])

    # read cache


    # print(data)
    return data, out_feats, text

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
# from dataset_utils.dataset import get_dataset
import numpy as np
import torch
from tqdm import tqdm
import random

def build_subgraph(data, node_idx_list, hop, max_size=204800, device="cuda:0"):
    subdata_list = []
    # print("Building subgraph...")
    # print(node_idx_list)
    for i in node_idx_list:
        # print("node id", i)
        data_subgraphs = k_hop_subgraph(node_idx=[i], num_hops=hop, edge_index=data.edge_index)
        # data_subgraphs = k_hop_subgraph(node_idx=[i], num_hops=hop, num_nodes=, edge_index=data.edge_index)
        subset = data_subgraphs[0]
        # print(subset)
        # if len(subset) > max_size:
        #     indices = torch.randperm(len(subset))[:max_size]
        #     subset = subset[indices]
        #     if i not in subset:
        #         subset[0] = i
        subdata = data.subgraph(subset)
        subdata = Data(x=subdata.x, edge_index=subdata.edge_index, y=data.y[i])
        subdata_list.append(subdata)
    return subdata_list


def build_subgraph_new(data, node_idx_list, hop, max_size=204800, device="cuda:0"):
    subdata_list = []
    # print("Building subgraph...")
    # print(node_idx_list)
    node_count = data.train_mask.shape[0]
    for i in node_idx_list:
        data_subgraphs = k_hop_subgraph(node_idx=[i], num_hops=hop, edge_index=data.edge_index, relabel_nodes=False)
        subdata = Data(x=data.x[data_subgraphs[0]], edge_index=data_subgraphs[1], y=data.y[i]).to(device)
        
        # print(len(subdata.x))
        # reorder node index
        node_idx = torch.zeros(node_count, dtype=torch.long, device=device)
        node_idx[data_subgraphs[0]] = torch.arange(len(data_subgraphs[0]), device=device)
        edge_index = torch.cat([node_idx[subdata.edge_index[0]].unsqueeze(0), node_idx[subdata.edge_index[1]].unsqueeze(0)])
        subdata.edge_index = edge_index
        
        subdata_list.append(subdata)
    return subdata_list