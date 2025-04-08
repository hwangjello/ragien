import torch
import json

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element: token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def filter_artwork_by_id(artwork_id, src_file='./data/paper_train_data.json'):
    with open(src_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    filtered_data = [item for item in data if item.get('id') == artwork_id]
    return json.dumps(filtered_data, ensure_ascii=False)
