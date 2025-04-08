import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from models.load_w2v import load_w2v_model

w2v_model = load_w2v_model()
stop_words = set(stopwords.words('english'))

def wmd_preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

def wmd_score(ref, out):
    ref_token = wmd_preprocess(ref)
    out_token = wmd_preprocess(out)
    return round(w2v_model.wmdistance(ref_token, out_token), 4)
