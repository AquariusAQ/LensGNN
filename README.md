# EnsemGNN
EnsemGNN: Ensembled Graph Large Language Model

## 0. Setup

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam dgl
pip install transformers
```

## 1. Download datasets and pre-trained model

### 1.1. Datasets
| Dataset      | Download |
|--------------|-------------|
| Cora         | Dataset download: [link](https://drive.google.com/file/d/1hxE0OPR7VLEHesr48WisynuoNMhXJbpl/view) | 
| Pubmed       | Dataset download: [link](https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view) |
| ogb-arxiv    | OGB page: [link](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)   |

All datasets need to be decompressed to `/GNN/datasets/[dataset_name]/[dataset_name] _orig`

### 1.2. Pre-trained language model

We employ Sentence-BERT as the language encoder for node text. Hugging Face: [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). To mitigate the impact of network fluctuations, we save the LM checkpoint to `/GNN/pretrained_lm`.

### 1.3. Pre-trained Large language model

We mainly utilize the open-source Baichuan2-13B-Chat as the LLM for experiment. HuggingFace: [link](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat). The checkpoint of this LLM is saved to the `/LLM` directory.

Our model involves modifications to the LLM, and the modified Baichuan2-13B model is located in the file `./LLM/Baichuan2-13B-Chat/modeling_baichuan.py`.

<!-- Additionally, we tested two LLMs: Falcon-7B and InternLM2.5-7B-chat; please refer to [here](https://huggingface.co/tiiuae/falcon-7b) and [here](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat) respectively. -->

## 2. Aligning multi-GNNs

- Initially, navigate to the `/GNN` directory.

- Subsequently, execute the train.py script, an example of a training command is as follows:

```bash
python train.py --gnn_dataset ogbn_arxiv \
    --train_type default \
    --epochs_encoders 1000 \
    --epochs_decoder  800 \
    --gnn_model gcn gat gin \
    --gnn_output_feature_size 5120 \
    --layer 2 \
    --hidden_channels 256 256 \
    --learning_rate_encoders 1e-2 1e-2 4e-3 \
    --learning_rate_linear 5e-4 5e-4 5e-4 \
    --weight_decay 0 \
    --dropout 0.2 \
    --model_device cuda:0
```

- Finally, the training settings, logs, and parameters of the GNN are saved to /GNN/out.

## 3. Generate dataset and training script for LLM

### 3.0. LLaMA-Factory

We use LLaMA-Factory as the scaffolding for fine-tuning the LLM. For the installation of LLaMA-Factory, please refer to its GitHub repository page: [link](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#getting-started ).

### 3.1. Setup for LLM fine-tuning

- Execute the main.py script, an example of the command is as follows:

```bash
use_dataset="ognb_arxiv"
epoch="3"
python main.py \
    --use_dataset $use_dataset \
    --epoch $epoch  \
    --train_type default \
    --graph_token_shape 1 \
    --device cuda:0 \
```

- After running the command, the LLM training settings will be saved to `./LLaMA-Factory/settings`, the LLM training script will be saved to `./LLaMA-Factory`, the LLM training dataset will be saved to `./LLaMA-Factory/data`, and the dataset_info.json will correspondingly be updated with new datasets. Simultaneously, all training settings will be saved to `./output_config`. The correspondence between the training samples of LLM and GNN representations is saved in the folder `./feature-cache`.

## 4. LLM Fine-tuning and Inference

- Execute the training script located in `./LLaMA-Factory`.These scripts end with "_train.sh". For the 16FP precision LoRA fine-tuning of the Baichuan2-13B model, approximately 32GB of GPU memory is typically required.

- Execute the inference script located in `./LLaMA-Factory`. Depending on the dataset used for testing the fine-tuned model, the test scripts end with "_predict_on_train.sh" or "_predict_on_test.sh".

- If you need to obtain the accuracy of the inference, you can run `evaluate_result.ipynb`, and the accuracy will be saved to the training settings json file in the `/output_config` directory.