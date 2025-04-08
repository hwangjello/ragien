from transformers import AutoTokenizer, AutoModel

def load_bert_model(model_name="intfloat/multilingual-e5-large-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
