##############################
#           main.py          #
##############################

import os
import torch
import math
import json
import ast
from datetime import datetime
from tqdm import tqdm
curr_time = datetime.now()
timestamp = datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
print("Now:", timestamp)

import argparse

parser = argparse.ArgumentParser(description='Process config.')

# main config
parser.add_argument('--tag', type=str, default="edgllam_1", help='train tag')
parser.add_argument('--use_dataset', type=str, default="cora", choices=["cora", "pubmed", "ogbn_arxiv", "ogbn_products", 'wikics'], help='use_dataset')
parser.add_argument('--train_type', type=str, default="subgraph", choices=['default','subgraph'], help='train_type')
parser.add_argument('--with_neighbor', type=int, default=1, help='with_neighbor')
parser.add_argument('--with_gnn_feature', nargs='*', type=str, default=["gcn","gat","gin"], help='with_gnn_feature list')
parser.add_argument('--graph_token_shape', nargs='+', type=int, default=[8], help='graph_token_shape list')
parser.add_argument('--essemble_feature', type=ast.literal_eval, default=True, help='essemble_feature')
parser.add_argument('--max_token', type=int, default=2047, help='max_token')
parser.add_argument('--with_text', type=ast.literal_eval, default=True, help='with_text')
parser.add_argument('--lm', type=str, default="embed", help='lm')

parser.add_argument('--use_token_cache', type=int, default=0, help="use_token_cache")

parser.add_argument('--top_k_token', type=int, default=1, help="top_k_token")

parser.add_argument('--use_kmeans', type=int, default=0, help="use_kmeans")


parser.add_argument('--max_test_sample', type=int, default=51200, help='max_test_sample')

# train config 
parser.add_argument('--epoch', type=int, default=5, help='llm train epoch')
parser.add_argument('--lr', type=str, default="2.0e-5", help='llm train lr')
parser.add_argument('--save_steps', type=int, default=500, help='llm train save_steps')

# script config
parser.add_argument('--embedding_size_pre_token', type=int, default=5120, help='llm embedding_size_pre_token')
parser.add_argument('--device', type=str, default="cuda:4", help='main device')
parser.add_argument('--device_subgraph', type=str, default="cuda:4", help='device_subgraph')
parser.add_argument('--cache_part_size', type=int, default=4096, help='cache_part_size')

args = parser.parse_args()
config = vars(args)
print(config)


config["name"] = config["tag"]
config["name"] += "_" + config["use_dataset"]
for gnn_name in config["with_gnn_feature"]:
    config["name"] += "_" + gnn_name
if len(config["with_gnn_feature"]) > 1:
    config["name"] += "_essembled" if config["essemble_feature"] else "_dispersed"
config["name"] += "_layer_" + "_".join([str(g) for g in config["graph_token_shape"]])
if config["with_text"] is True:
    config["name"] += "_with_{}_neighbor".format(config["with_neighbor"]) #if config["with_neighbor"] else "_without_neighbor"
else:
    config["name"] += "_without_text"
# config["name"] += "_with_label" if config["with_label"] else "_without_label"

print("name:",config["name"])

embedding_size_pre_token = config["embedding_size_pre_token"]
device = config["device"]

cache_roots = [
    "./cache",
    "./cache/feature",
    "./cache/token",
    "./cache/llm_dataset"
]
for cache_root in cache_roots:
    if not os.path.exists(cache_root):
        os.mkdir(cache_root)

# read dataset
from GNN.dataset_utils.dataset import get_dataset
data, out_feats_size, texts = get_dataset(dataset=config["use_dataset"], lm=config["lm"], enable_tokenize_cache=True, seed=0,  root="./GNN", device=config["device_subgraph"])
data = data.to(config["device_subgraph"])
data_size = data.train_mask.shape[0]
# data_size = 2449029


if config["train_type"] == "subgraph":
    feature_cache_root = "./cache/feature/{}-{}{}-{}".format(config['use_dataset'], '-'.join(config['with_gnn_feature']), "" if config["essemble_feature"] else "-dispersed",\
                                                       "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))
    token_cache_root = "./cache/token/{}-{}{}-{}".format(config['use_dataset'], '-'.join(config['with_gnn_feature']), "" if config["essemble_feature"] else "-dispersed",\
                                                           "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))



    #########################################
    # 1. GNN Encoder feature #
    #########################################
    if len(config["with_gnn_feature"]) > 0:
        # if not os.path.exists(feature_cache_root):
        print("Encoding...")

        # read encoder list
        gnn_encoder_list = []
        gnn_encoder_root_list = []
        if config["essemble_feature"]:
            gnn_encoder_root = "./GNN/out/{}-{}-{}".format(config['use_dataset'], '-'.join(config['with_gnn_feature']), \
                                                           "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))
            gnn_encoder_root_list.append(gnn_encoder_root)
        else:
            for gnn_name in config["with_gnn_feature"]:
                gnn_encoder_root = "./GNN/out/{}-{}-{}".format(config['use_dataset'], gnn_name, \
                                                               "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))
                gnn_encoder_root_list.append(gnn_encoder_root)
        print(gnn_encoder_root_list)

        # read encoder
        from utils import read_gnn_encoder
        for gnn_encoder_root in gnn_encoder_root_list:
            gnn_encoder_list.append(read_gnn_encoder(gnn_encoder_root, config, device))
        print(gnn_encoder_list)

        # encode
        from utils import encoding
        for cache_i in range(math.ceil(data_size / config["cache_part_size"])):
            if config['use_token_cache'] == 1 and os.path.exists(os.path.join(token_cache_root, "token_{:0>4}.pt".format(cache_i))):
                # print(os.path.join(token_cache_root, "token_{:0>4}.pt".format(cache_i)), "exists, skipping")
                continue
            
            if os.path.exists(os.path.join(feature_cache_root, "feature_{:0>4}.pt".format(cache_i))):
                print(os.path.join(feature_cache_root, "feature_{:0>4}.pt".format(cache_i)), "exists, skipping")
                continue

            print(os.path.join(feature_cache_root, "feature_{:0>4}.pt".format(cache_i)), "not found, encoding...")

            feature = []
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated(4))
            for i, gnn_name in enumerate(config["with_gnn_feature"]):
                if config["essemble_feature"]:
                    encoder = gnn_encoder_list[0][0][i]
                    gnn_config = gnn_encoder_list[0][1]
                else:
                    encoder = gnn_encoder_list[i][0][0]
                    gnn_config = gnn_encoder_list[i][1]
                encode_range = (cache_i*config["cache_part_size"], min(data_size, (cache_i+1)*config["cache_part_size"]))
                feature.append(encoding(data, encoder, encode_range, config=config, gnn_config=gnn_config, device=device, device_subgraph=config["device_subgraph"]))
            print(feature[0])
            # save to cache
            if not os.path.exists(feature_cache_root):
                os.mkdir(feature_cache_root)
            torch.save(feature, os.path.join(feature_cache_root, "feature_{:0>4}.pt".format(cache_i)))
            print("Cache saved to", feature_cache_root)



        encode_detail = {
            "config": config,
            "data_size": data_size
        }
        with open(os.path.join(feature_cache_root, "encode_detail.json"), "w") as f:
            json.dump(encode_detail, f)
            f.close()


elif config["train_type"] == "default":
    feature_cache_root = "./cache/feature/{}-{}{}-{}-graph".format(config['use_dataset'], '-'.join(config['with_gnn_feature']), "" if config["essemble_feature"] else "-dispersed",\
                                                       "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))
    token_cache_root = "./cache/token/{}-{}{}-{}-graph".format(config['use_dataset'], '-'.join(config['with_gnn_feature']), "" if config["essemble_feature"] else "-dispersed",\
                                                           "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))


    if len(config["with_gnn_feature"]) > 0:
        # if not os.path.exists(feature_cache_root):
        print("Encoding...")

        # read encoder list
        gnn_encoder_list = []
        gnn_encoder_root_list = []
        if config["essemble_feature"]:
            gnn_encoder_root = "./GNN/out/{}-{}-{}-graph".format(config['use_dataset'], '-'.join(config['with_gnn_feature']), \
                                                           "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))
            gnn_encoder_root_list.append(gnn_encoder_root)
        else:
            for gnn_name in config["with_gnn_feature"]:
                gnn_encoder_root = "./GNN/out/{}-{}-{}-graph".format(config['use_dataset'], gnn_name, \
                                                               "-".join([str(embedding_size_pre_token*token) for token in config["graph_token_shape"]]))
                gnn_encoder_root_list.append(gnn_encoder_root)
        print(gnn_encoder_root_list)

        # read encoder
        from utils import read_gnn_encoder
        for gnn_encoder_root in gnn_encoder_root_list:
            gnn_encoder_list.append(read_gnn_encoder(gnn_encoder_root, config, device))
        print(gnn_encoder_list)

        # encode
        from utils import encoding_graph
        # for cache_i in range(math.ceil(data_size / config["cache_part_size"])):
        if config['use_token_cache'] == 1 and os.path.exists(os.path.join(token_cache_root, "token.pt")):
            # print(os.path.join(token_cache_root, "token_{:0>4}.pt".format(cache_i)), "exists, skipping")
            pass
            
        elif os.path.exists(os.path.join(feature_cache_root, "feature.pt")):
            print(os.path.join(feature_cache_root, "feature.pt"), "exists, skipping")
        
        else:
            print(os.path.join(feature_cache_root, "feature.pt"), "not found, encoding...")

            feature = []
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated(4))
            for i, gnn_name in enumerate(config["with_gnn_feature"]):
                if config["essemble_feature"]:
                    encoder = gnn_encoder_list[0][0][i]
                    gnn_config = gnn_encoder_list[0][1]
                else:
                    encoder = gnn_encoder_list[i][0][0]
                    gnn_config = gnn_encoder_list[i][1]
                feature.append(encoding_graph(data, encoder, config=config, gnn_config=gnn_config, device=device, device_subgraph=config["device_subgraph"]))
            print(feature[0])
            # save to cache
            if not os.path.exists(feature_cache_root):
                os.mkdir(feature_cache_root)
            torch.save(feature, os.path.join(feature_cache_root, "feature.pt"))
            print("Cache saved to", feature_cache_root)



        encode_detail = {
            "config": config,
            "data_size": data_size
        }
        with open(os.path.join(feature_cache_root, "encode_detail.json"), "w") as f:
            json.dump(encode_detail, f)
            f.close()

else:
    exit(0)

#####################
# 3.  llm dataset #
#####################

# 构建 gnn 数据集
if not os.path.exists("./llm_dataset"):
    os.mkdir("./llm_dataset")

if len(config["with_gnn_feature"]) > 0:
    with open(os.path.join(feature_cache_root, "encode_detail.json")) as f:
            unembedding_detail = json.load(f)
            f.close()
            data_size = unembedding_detail["data_size"]
    # print(data_size)

gnn_features = []
if len(config["with_gnn_feature"]) > 0:
    if config["train_type"] == "subgraph":
        for cache_i in tqdm(range(math.ceil(data_size / config["cache_part_size"]))):
            token = torch.load(os.path.join(feature_cache_root, "feature_{:0>4}.pt".format(cache_i))).cpu()
            gnn_features.append(token)
        gnn_features = torch.cat(gnn_features, dim=1).to(config["device"])
    elif config["train_type"] == "default":
        gnn_features = torch.cat(torch.load(os.path.join(feature_cache_root, "feature.pt")))
        # print(gnn_features)
        # exit(0)
    print("Token size:", gnn_features.shape)

from llm_dataset_builder import build_llm_dataset
train_dataset_path, test_dataset_path, train_label_path, test_label_path = build_llm_dataset(gnn_features, config, varbose=True)



##############################
# 4. LLM training bash #
##############################


import shutil
for file_path in [train_dataset_path, test_dataset_path]:
    shutil.copy(file_path, os.path.join("./LLaMA-Factory/data/", file_path.split("/")[-1]))

import json
llm_dataset_info_path = "./LLaMA-Factory/data/dataset_info.json"
with open(llm_dataset_info_path) as f:
    llm_dataset_info = json.load(f)
flag = "gnn_aggregator_"+config["name"]
llm_dataset_info[flag+"_train"] = {}
llm_dataset_info[flag+"_train"]["file_name"] = train_dataset_path.split("/")[-1]
llm_dataset_info[flag+"_test"] = {}
llm_dataset_info[flag+"_test"]["file_name"] = test_dataset_path.split("/")[-1]
with open(llm_dataset_info_path, "w") as f:
    json.dump(llm_dataset_info, f, indent=4)


setting_root = "./LLaMA-Factory/settings"
if not os.path.exists(setting_root):
    os.mkdir(setting_root)

train_setting = \
f"""# {flag}_train.yaml \\
### model
model_name_or_path: ../LLM/Baichuan2-13B-Chat

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_dropout: 0.1
loraplus_lr_ratio: 16
# use_rslora: true
# use_dora: true
# pissa_init: true

### dataset
dataset: {flag}_train
template: baichuan2
cutoff_len: {config["max_token"]}
max_samples: 524288
overwrite_cache: false
preprocessing_num_workers: 16

### output
output_dir: saves/Baichuan2-13B-Chat/lora/sft/{flag}
logging_steps: 10
save_steps: {config["save_steps"]}
plot_loss: true
overwrite_output_dir: false

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: {config["lr"]}
num_train_epochs: {config["epoch"]}
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
# flash_attn: fa2

## eval
val_size: 0.1
per_device_eval_batch_size: 4
evaluation_strategy: steps
eval_steps: 200

"""
with open(os.path.join(setting_root, f"{flag}_train.yaml"), "w") as f:
    f.write(train_setting)
    f.close()



# (on train)
predict_script = \
f"""# {flag}_predict_on_test.yaml \\
### model
model_name_or_path: ../LLM/Baichuan2-13B-Chat
adapter_name_or_path: saves/Baichuan2-13B-Chat/lora/sft/{flag}

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: {flag}_train
template: baichuan2
cutoff_len: {config["max_token"]}
max_samples: 1024
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/Baichuan2-13B-Chat/lora/predict/{flag}_predict_on_train
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 4
predict_with_generate: true
"""
with open(os.path.join(setting_root, f"{flag}_predict_on_train.yaml"), "w") as f:
    f.write(predict_script)
    f.close()


# (on test)
predict_script = \
f"""# {flag}_predict_on_test.yaml \\
### model
model_name_or_path: ../LLM/Baichuan2-13B-Chat
adapter_name_or_path: saves/Baichuan2-13B-Chat/lora/sft/{flag}

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: {flag}_test
template: baichuan2
cutoff_len: {config["max_token"]}
max_samples: 51200
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/Baichuan2-13B-Chat/lora/predict/{flag}_predict_on_test
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 4
predict_with_generate: true
"""
with open(os.path.join(setting_root, f"{flag}_predict_on_test.yaml"), "w") as f:
    f.write(predict_script)
    f.close()


train_setting = \
"""#\\
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train {}
""".format(os.path.join("./settings", f"{flag}_train.yaml"))
with open(os.path.join("./LLaMA-Factory", f"{flag}_train.sh"), "w") as f:
    f.write(train_setting)
    f.close()

predict_script = \
"""#\\
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train {}
""".format(os.path.join("./settings", f"{flag}_predict_on_train.yaml"))
with open(os.path.join("./LLaMA-Factory", f"{flag}_predict_on_train.sh"), "w") as f:
    f.write(predict_script)
    f.close()


predict_script = \
"""#\\
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train {}
""".format(os.path.join("./settings", f"{flag}_predict_on_test.yaml"))
with open(os.path.join("./LLaMA-Factory", f"{flag}_predict_on_test.sh"), "w") as f:
    f.write(predict_script)
    f.close()


# save config
if not os.path.exists("./output_config"):
    os.mkdir("./output_config")
output_config_path = os.path.join("./output_config", config["name"]+".json")
with open(output_config_path, "w") as f:
    json.dump({
            "config": config,
            "data_size": data_size
        }, f, indent=4)