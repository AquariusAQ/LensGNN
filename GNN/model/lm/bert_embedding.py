import torch
import os
from transformers import AutoTokenizer, AutoModel
import math
from tqdm import tqdm




# inputs = tokenizer('我是一个好人', return_tensors='pt')
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print('last_hidden_states:' ,last_hidden_states.shape)
# pooler_output = outputs.pooler_output
# print('pooler_output: ', pooler_output.squeeze(0).shape)

def embed(texts, max_length=512, enable_cache=False, cache_root="", device="cuda:0"):
    
    cache_path = os.path.join(cache_root, 'feature_embedding.pt')
    print("cache_path:", cache_path)
    if enable_cache and cache_root == "":
        print("cache_root can not be empty when enable_cache is True!")
    if not enable_cache or not os.path.exists(cache_path):
        print("Cache disabled or find no cache")
        if enable_cache and not os.path.exists(cache_root):
            os.mkdir(cache_root)

        model_id = 'pretrained_lm/paraphrase-multilingual-MiniLM-L12-v2'
        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.eval()
        model = model.to(device)

        batch_size = 32
        feature_list = []
        with torch.no_grad():
            # for i in tqdm(range(0, math.ceil(len(texts) / batch_size))):
            for i in range(0, math.ceil(len(texts) / batch_size)):
                text_part = texts[i*batch_size: min((i+1)*batch_size, len(texts))]
                feature_tokenized = tokenizer(text_part, padding='max_length', return_tensors='pt', max_length=max_length, truncation=True).to(device)
                outputs = model(**feature_tokenized)
                # print(outputs.last_hidden_state.shape, outputs.pooler_output.shape)
                feature_embedding = outputs.pooler_output.to("cpu")
                feature_list.append(feature_embedding)
            feature_embedding = torch.cat(feature_list, dim=0).to("cpu")

        if enable_cache:
            print("Saving cache...")
            torch.save(feature_embedding, cache_path)
    else:
        print("Find cache:", cache_path)
        feature_embedding = torch.load(cache_path)

    print("feature shape:", feature_embedding.shape)
    return feature_embedding.to("cpu").numpy()
