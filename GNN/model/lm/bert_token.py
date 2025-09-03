import torch
import os
from transformers import AutoTokenizer, BertModel

device = "cuda"
# 仅使用 Bert Tokenizer 处理原文本（分词）
def tokenize(texts, max_length=512, enable_cache=False, cache_root=""):
    cache_path = os.path.join(cache_root, 'feature_tokenized.pt')
    if enable_cache and cache_root == "":
        print("cache_root can not be empty when enable_cache is True!")
    if not enable_cache or not os.path.exists(cache_path):
        print("Cache disabled or find no cache")
        if not os.path.exists(cache_root):
            os.mkdir(cache_root)
        tokenizer = AutoTokenizer.from_pretrained('pretrained_lm/paraphrase-multilingual-MiniLM-L12-v2')#.to(device)
        texts = texts#.to(device)
        feature_tokenized = tokenizer(texts, padding='max_length', max_length=max_length, truncation=True)
        
        if enable_cache:
            print("Saving cache...")
            torch.save(feature_tokenized, cache_path)
    else:
        print("Find cache:", cache_path)
        feature_tokenized = torch.load(cache_path)

    return feature_tokenized["input_ids"]
