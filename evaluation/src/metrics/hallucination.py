import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def make_noun(text):
    text = re.sub(r'[";!@~`$%^&*=+_:/.,><‘’]\'\\-', '', text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('N')]
    fw = [word for word, pos in tagged if pos.startswith('F')]
    cd = [word for word, pos in tagged if pos.startswith('CD')]
    return list(set(nouns + fw + cd))

def nonOri(text1, text2):
    return list(set(text1) - set(text2))

def compute_hal_score(ground_truth, answer, source):
    ref_decom = make_noun(ground_truth)
    out_decom = make_noun(answer)
    src_decom = make_noun(source)
    hal_cand = nonOri(out_decom, ref_decom)
    hal_fin = nonOri(hal_cand, src_decom)
    return 1 - len(hal_fin) / len(out_decom) if len(out_decom) > 0 else 0.0
