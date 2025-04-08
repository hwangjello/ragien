import torch
import torch.nn.functional as F
from utils.preprocessing import mean_pooling
from models.load_bert import load_bert_model
from models.load_sbert import load_sbert_model

bert_tokenizer, bert_model = load_bert_model()
sbert_model = load_sbert_model()

def get_bert_score(sentences):
    encoded = bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = bert_model(**encoded)
    sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])
    return F.cosine_similarity(
        sentence_embeddings[0].unsqueeze(0),
        sentence_embeddings[1].unsqueeze(0)
    ).item()

def get_sbert_score(sentences):
    embeddings = sbert_model.encode(sentences)
    return F.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0)
    ).item()
