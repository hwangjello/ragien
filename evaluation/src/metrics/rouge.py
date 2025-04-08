from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

class NLTKTokenizer:
    def tokenize(self, text):
        return word_tokenize(text)

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=NLTKTokenizer()
)

def rouge_l(ref, out):
    rouge_score = scorer.score(ref, out)['rougeL']
    return round(rouge_score.fmeasure, 4)
