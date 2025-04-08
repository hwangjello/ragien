from sentence_transformers import SentenceTransformer

def load_sbert_model(model_name="all-mpnet-base-v2"):
    return SentenceTransformer(model_name)
