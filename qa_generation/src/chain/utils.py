from langchain.output_parsers import ResponseSchema
import json
from typing import Literal
import re
# import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import os
import random


def generate_response_schemas(n):
    schemas = []
    for i in range(1, n + 1):
        schemas.append(ResponseSchema(name=f"Q{i}", description=f"Content of Question {i}"))
        schemas.append(ResponseSchema(name=f"A{i}", description=f"Content of Answer {i}"))
    return schemas

def generate_json_format(n):
    data = {}
    for i in range(1, n + 1):
        data[f"Q{i}"] = f"Content of Question {i}"
        data[f"A{i}"] = f"Content of Answer {i}"
    inner_json = json.dumps(data, ensure_ascii=False, indent=2)
    return "{" + inner_json + "}"

def make_noun(text):
    text = re.sub(r'[";!@~`$%^&*=+_:/.,><‘’]\'\-', '', text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('N')]
    fw = [word for word, pos in tagged if pos.startswith('F')]
    res = nouns + fw
    res = list(set(res))

    return res

def nonOri(text1, text2):
    res = list(set(text1) - set(text2))
    return res

def hal_score(formatted_qa, context, n):
    answers = []
    scores = []

    context_decom = make_noun(context)

    for i in range(1, n+1):
        key = f"A{i}"
        if key in formatted_qa:
            answers.append(formatted_qa[key])

    for ans in answers:
        ans_decom = make_noun(ans)
        hal_cand = nonOri(ans_decom, context_decom)
        score = 1-len(hal_cand)/len(ans_decom)

        scores.append(score)
    return scores

def is_valid_qa(formatted_qa):
    for key, value in formatted_qa.items():
        if value == '':
            print(f"Value of field '{key}' is empty.")
            return False
    return True

# 새로운 qa 데이터를 추가하는 함수
def save_qa_to_file(new_data, file_name):
    # 파일이 이미 존재하면 기존 데이터를 읽어옴
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  # 파일이 비어있다면 빈 리스트로 초기화
    else:
        existing_data = []  # 파일이 없으면 빈 리스트로 초기화

    # 새로운 데이터를 기존 리스트에 추가
    existing_data.append(new_data)

    # 데이터를 파일에 저장
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)