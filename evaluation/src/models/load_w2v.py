import gensim.downloader as api

def load_w2v_model(model_name='word2vec-google-news-300'):
    return api.load(model_name)
