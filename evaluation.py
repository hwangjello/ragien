
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# WMD 데이터
from gensim.models import KeyedVectors
import gensim.downloader as api

from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F



# Word2Vec 모델 로드
w2v_model = api.load('word2vec-google-news-300')

nltk.download('punkt')
nltk.download('stopwords')

#BERT 모델 로드
bert_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
bert_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# SentenceTransformer 모델 로드
sbert_model = SentenceTransformer("all-mpnet-base-v2")

# 작품번호를 받아올 원본 파일
src_file = './data/paper_train_data.json'
# 작품번호별 질문 정답 모델의 출력을 리스트화 한 파일 - json 파일이어야 함. ready_for_wmd에서 생성된 txt 파일을 json으로 확장자 변경
listed_file = "./result/listed_qao_hal_score.json"
# 점수 완료 후 저장될 csv 파일. - 의미 없음.
csv_output = "./result/sbert_score_qao_score_comparison.csv"
# 점수 완료 후 저장될 json 파일 - 중요.
json_output = "./result/sbert_score_qao_score_comparison.json"

import json

def filter_artwork_by_id(artwork_id):
    # JSON 파일을 UTF-8 인코딩으로 읽어옵니다.
    with open(src_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # data가 리스트 형태로 구성되어 있다고 가정하고, 각 항목의 '작품번호'가 artwork_id와 일치하는지 확인합니다.
    filtered_data = [item for item in data if item.get('id') == artwork_id]
    filtered_data = json.dumps(filtered_data, ensure_ascii=False)

    return filtered_data



def make_noun(text):
    text = re.sub(r'[";!@~`$%^&*=+_:/.,><‘’]\'\-', '', text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('N')]
    fw = [word for word, pos in tagged if pos.startswith('F')]
    cd = [word for word, pos in tagged if pos.startswith('CD')]
    res = nouns + fw + cd
    res = list(set(res))

    return res

def nonOri(text1, text2):
    res = list(set(text1) - set(text2))
    return res

# BLEU Score
from nltk.translate.bleu_score import sentence_bleu

def bleu_n(ref,out):
    ref = [ref.split()]
    out = out.split()
    bleu_ngram = sentence_bleu(ref, out)
    return round(bleu_ngram, 4)

# ROUGE-L Score
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

def rouge_l(ref,out):
    rouge_score = scorer.score(ref, out)['rougeL']
    return round(rouge_score.fmeasure, 4)

# WMD
def wmd_preprocess(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'\W', ' ', text)  # 특수 문자 제거
    tokens = word_tokenize(text)  # 토큰화
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # 불용어 제거
    return tokens

def wmd_score(ref,out):
    ref_token = wmd_preprocess(ref)
    out_token = wmd_preprocess(out)
    wmd_score = w2v_model.wmdistance(ref_token,out_token)
    return round(wmd_score, 4)


# BERT Score - 코드 내부 실행
# SBERT Score - 코드 내부 실행
# Hallucination Score - 코드 내부 실행

import json
import pandas as pd

# JSON 파일 로드 (파일 경로에 맞게 수정)
with open(listed_file, "r", encoding="utf-8") as f:
    data_json = json.load(f)

results = []

# JSON 파일에 저장된 각 작품번호별 데이터 처리
for artwork_id, content in data_json.items():

    questions = content["question"]
    ground_truths = content["ground_truth"]
    answers = content["answer"]
    source = filter_artwork_by_id(artwork_id[5:])

    for idx, (question, ground_truth, answer) in enumerate(zip(questions, ground_truths, answers)):
        sentences = [ground_truth, answer]

        # BLEU
        bleu = bleu_n(ground_truth,answer)

        # rouge-l
        rouge = rouge_l(ground_truth,answer)

        # wmd
        wmd = wmd_score(ground_truth,answer)

        # bert
        encoded = bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = bert_model(**encoded)
        sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])

        bert = F.cosine_similarity(
            sentence_embeddings[0].unsqueeze(0),
            sentence_embeddings[1].unsqueeze(0)
        ).item()

        # sbert
        embeddings = sbert_model.encode(sentences)

        sbert = torch.nn.functional.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        ).item()

        # hal_score
        ref_decom = make_noun(ground_truth)
        out_decom = make_noun(answer)
        src_decom = make_noun(source)
        hal_cand = nonOri(out_decom,ref_decom)
        hal_fin = nonOri(hal_cand,src_decom)
        hal_score = 1- len(hal_fin)/len(out_decom)

        if sbert > 0 and hal_score > 0:
            fin_score = 2.0 / ((1.0 / sbert) + (1.0 / hal_score))
        else:
            fin_score = 0.0

        results.append([artwork_id, idx + 1, question, answer, ground_truth, bleu, rouge, wmd, bert, sbert, hal_score, fin_score])
        print(f"{artwork_id} - {idx+1}/{len(questions)} sbert score: {sbert}, hallucination score: {hal_score} -> final score: {fin_score:.4f}")


df_results = pd.DataFrame(results, columns=["paper_id", "Question_Number", "Question", "Answer", "Ground Truth","BLEU", "ROUGE-L", "WMD", "BERT", "SBERT", "Hal_score","Tot_score"])

df_results.to_csv(csv_output, index=False, encoding="utf-8")

print(f"Evaluation Completed! The result saved in '{csv_output}'.")

import json

df_results.to_json(json_output, orient='records', force_ascii=False, indent=4)
print(f"Evaluation Completed! The result saved in '{json_output}'.")
