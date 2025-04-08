from nltk.translate.bleu_score import sentence_bleu

def bleu_n(ref, out):
    ref = [ref.split()]
    out = out.split()
    return round(sentence_bleu(ref, out), 4)