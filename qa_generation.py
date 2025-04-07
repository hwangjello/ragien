

from langchain_community.llms import Ollama

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")

# JSON 데이터를 저장할 파일 이름 - 지금은 테스트 파일. 나중에 _test 빼기
file_name = "./paper_train_qa.json"
data_name = './paper_train_data.json'
# JSON 파일 읽기
with open(data_name, 'r', encoding='utf-8') as f:
    data = json.load(f)


"""# 프롬프트 메이커"""

requirements = """
first line of qa-pair format must be "artwork_num", such as
Artwork Number : "artwork_num",
Q : question
A : answer
...
\n
Each 'Question' must contains full 'title' of the paper and "first_author' from given context at the same time. - Very important \n
"""

qa_num =5
example = """
Q: According to 'Unifying Adversarial Training Algorithms with Flexible Deep Data Gradient Regularization' by Alexander G. Ororbia II, what is the DataGrad framework viewed as?
"""

"""## N개 세트 메이커"""

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
from typing import Literal
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import os
import random





template = '''
You are a language model responsible for generating high-quality prompt to make Question-Answer (QA) pairs that precisely satisfy the user's requirements.
Simply you are prompt generator.
Generate prompt for AI model that can satisfies following specifications and requirements provided by the user:

Requirements:
1. Logically include all of the user's specified requirements in each QA pair. \n
2. Follow the specified QA pair format shown below precisely. \n
3. If the user has provided example QA pairs, clearly reference and mimic their style and format. \n

\n\n

Add 1 : You are NOT the qa-set generator. You must provide only prompt. Don't generate QA-SET by yourself. \n
Add 2 : You must never create examples arbitrarily on your own. \n
Add 3 : Do not make other passages excepts prompt. \n
\n\n

Format of your generation. You must follow provided prompt format provided below. Do not modify Format part of given format by your own.
\n
Prompt Format
------------------------------
QA-Pair Format:
Q1: "Write the first question strictly based on the paragraph."
A1: "Write the first answer strictly based on the paragraph."
...
Q{num}: "Write the {num}-th question strictly based on the paragraph."
A{num}: "Write the {num}-th answer strictly based on the paragraph."


User-provided Requirements:
- (Explicitly list all user's requirements clearly and logically here.)

------------------------------
\n
":
\n\n

User Requirements : {user_req}


'''

prompt = ChatPromptTemplate.from_template(template)


# Chain
pc_think_chain = prompt | llm | StrOutputParser()


template = '''
You are a language model responsible for generating high-quality prompt to make Question-Answer (QA) pairs that precisely satisfy the user's requirements.
Simply you are prompt generator.
Generate prompt for AI model that can satisfies following specifications and requirements provided by the user:

Requirements:
1. Logically include all of the user's specified requirements in each QA pair. \n
2. Follow the specified QA pair format shown below precisely. \n
3. If the user has provided example QA pairs, clearly reference and mimic their style and format. \n

\n\n

Add 1 : You are NOT the qa-set generator. You must provide only prompt. Don't generate QA-SET by yourself. \n
Add 2 : You must never create examples arbitrarily on your own. \n
Add 3 : Do not make other passages excepts prompt. \n
\n\n

Format of your generation. You must follow provided prompt format provided below. Do not modify Format part of given format by your own.
\n
Prompt Format
------------------------------
QA-Pair Format:
Q1: "Write the first question strictly based on the paragraph."
A1: "Write the first answer strictly based on the paragraph."

User-provided Requirements:
- (Explicitly list all user's requirements clearly and logically here.)

------------------------------
\n
":
\n\n

User Requirements : {user_req}


'''

prompt = ChatPromptTemplate.from_template(template)


# Chain
pc_one_chain = prompt | llm | StrOutputParser()



"""# 예시 QA 제작 체인"""

# QA-pair Generation

template = '''
You are a language model responsible for generating high-quality Question-Answer (QA) pairs that precisely satisfy the user's requirements. \n
Generate exactly Only One QA pairs meeting all the following specifications and requirements provided by the user. \n

{prompt}

\n
":
{context}

\n\n
Add 1 : One question Must contain only one subject. \n
Add 2 : Every Question must strictly stick to the offered context. - priority high \n
Add 3 : Every QA-Set must be compsed with the vocabulary from the offered context. - First priority -double check \n
Add 4 : Every answer you produce must exclusively use the vocabulary, phrases, and sentences directly extracted from these documents. Do not create any new words or expressions that are not present in the provided retrieved documents. \n
Add 5 : All answers should be statements with clear causal relationships that answer the question. - second prioirity \n
Add 6 : {user_req} - first priority - Double Check \n
'''

prompt = ChatPromptTemplate.from_template(template)

# Chain
qa_think_one_chain = prompt | llm | StrOutputParser()


# qa_think_one = re.sub(r'<think>.*?</think>\n*', '', qa_think_one, flags=re.DOTALL)

# print(qa_think_one)



template = '''
You are a language model responsible for generating high-quality Question-Answer (QA) pairs that precisely satisfy the user's requirements. \n
Generate exactly {num} QA pairs meeting all the following specifications and requirements provided by the user. \n

\n\n

{prompt}
\n

Examples: {example}

\n
":
{context}

\n\n
Add 1 : One question Must contain only one subject. \n
Add 2 : Every Question must strictly stick to the offered context. - priority high \n
Add 3 : Every QA-Set must be compsed with the vocabulary from the offered context. - First priority -double check \n
Add 4 : Every answer you produce must exclusively use the vocabulary, phrases, and sentences directly extracted from these documents. Do not create any new words or expressions that are not present in the provided retrieved documents. \n
Add 5 : All answers should be statements with clear causal relationships that answer the question. - second prioirity \n
Add 6 : {user_req} - first priority - Double Check \n
Add 7 : Rethink That You met every requirements that user offered. \n
'''

prompt = ChatPromptTemplate.from_template(template)

# Chain
qa_think_chain = prompt | deep | StrOutputParser()


# import re
# qa_gen = re.sub(r'<think>.*?</think>\n*', '', qa_think, flags=re.DOTALL)

# print(qa_gen)


from langchain.output_parsers import ResponseSchema, StructuredOutputParser

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
    # 내부 JSON 문자열 생성 (한 쌍의 중괄호 포함)
    inner_json = json.dumps(data, ensure_ascii=False, indent=2)
    # 내부 문자열을 추가 중괄호로 감싸서 두 쌍의 중괄호 생성
    return "{" + inner_json + "}"




response_schemas = generate_response_schemas(qa_num)
json_format = generate_json_format(qa_num)

parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = '''
You are a formatting expert tasked with transforming provided question-answer pairs into a specific format. \n
Extract only the questions and answers from the given content and convert them precisely into the format provided below. \n
Do not include additional explanations or descriptions beyond the requested format. \n

The required format is as follows
{json}
":
{context}

'''

prompt = ChatPromptTemplate.from_template(template)

# Chain
format_chain = prompt | llm | parser


"""# 내용 평가"""

response_schemas = [
    ResponseSchema(name="binary_score", description="Is the Answer related to the given context, 'yes' or 'no'")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Prompt
system = """
You are a grader assessing whether an answer related to the given context. \n
It does not need to be a stringent test. The goal is to filter out the answers correlated with the given context. \n
Give a binary score 'yes' or 'no'. \n
'yes' means that the answer correlated with the given context. \n
Otherwise, return 'no' \n
The answer should be one of these.\n
{{
    "binary_score": "yes"
}}
or
{{
    "binary_score": "no"
}}
"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system+"\n{format_instructions}"),
        ("human", "Given Context: \n\n {context} \n\n Answer: {answer}"),
    ]
)
answer_prompt = answer_prompt.partial(format_instructions=parser.get_format_instructions())
answer_grader = answer_prompt | llm | parser


"""# 포맷 검사"""

response_schemas = [
    ResponseSchema(name="binary_score", description="Response adheres exactly to the given format, 'yes' or 'no'")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Prompt
system = """
You are a strict format validator. Your task is to ensure that the given response adheres exactly to the specified format. \n
Specifically, verify that each JSON field is properly populated. All responses must be provided in Korean. \n
The required format is as follows

{json}

'yes' means that the response adheres exactly to the given format. \n
Otherwise, return 'no' \n
The answer should be one of these.\n
{{
    "binary_score": "yes"
}}
or
{{
    "binary_score": "no"
}}
"""
format_verify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system+"\n{format_instructions}"),
        ("human", "Response: {answer}"),
    ]
)
format_verify_prompt = format_verify_prompt.partial(format_instructions=parser.get_format_instructions())
format_grader = format_verify_prompt | llm | parser




"""# 데이터 로드"""



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

"""# 비어있는 Value 검사"""

def is_valid_qa(formatted_qa):
    for key, value in formatted_qa.items():
        if value == '':
            print(f"필드 '{key}'의 값이 비어 있습니다.")
            return False
    return True

"""# 파일 저장"""



# JSON 객체 분리
if isinstance(data, list):
    papers = data  # 리스트 형태로 처리
else:
    papers = [data]

"""# QA 쌍 생성 체인"""



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

def hal_score(formatted_qa, context):
    answers = []
    scores = []

    context_decom = make_noun(context)

    for i in range(1, 4):
        key = f"A{i}"
        if key in formatted_qa:
            answers.append(formatted_qa[key])

    for ans in answers:
        ans_decom = make_noun(ans)
        hal_cand = nonOri(ans_decom, context_decom)
        score = 1-len(hal_cand)/len(ans_decom)

        scores.append(score)
    return scores



def ex_gen(papers, pc_one_chain, qa_think_one_chain, requirements=" "):
    paper = random.choice(papers)
    context = json.dumps(paper, ensure_ascii=False, indent=4)

    pc_one = pc_one_chain.invoke({"user_req": requirements})
    qa_think_one = qa_think_one_chain.invoke({"prompt":pc_one, "context": context, "user_req": requirements})
    ex_qa = re.sub(r'<think>.*?</think>\n*', '', qa_think_one, flags=re.DOTALL)
    return ex_qa

def qa_gen(papers, pc_think_chain, qa_think_chain, format_chain, answer_grader, format_grader, file_name, example=" ",requirements=" ", qa_num=5, threshold=0.8):
    # 각 작품에 대해 QA 생성
    for paper in papers:
        context = json.dumps(paper, ensure_ascii=False, indent=4)
        while True:
            try:
                pc = pc_think_chain.invoke({"user_req": requirements,"num":qa_num})
                qa_think = qa_think_chain.invoke({"num":qa_num,"prompt":pc,"example":example, "context": context, "user_req": requirements})
                qa_gen = re.sub(r'<think>.*?</think>\n*', '', qa_think, flags=re.DOTALL)
                json_format = generate_json_format(qa_num)
                formatted_qa = format_chain.invoke({"context": qa_gen,"json":json_format})
                qa_grade = answer_grader.invoke({"context": context,"answer": formatted_qa})
                format_grade = format_grader.invoke({"json":json_format, "answer": formatted_qa})
                is_valid = is_valid_qa(formatted_qa)
                scores = hal_score(formatted_qa, context)
                print(scores)
                if not all(score >= threshold for score in scores):
                    continue
                if qa_grade["binary_score"] == "yes" and format_grade["binary_score"] == "yes" and is_valid == True:
                    break
                else:
                    print("적절한 QA가 아닙니다. QA 재생성.")
                    continue

            except Exception as e:
                print(f"에러 발생: {e}. 다시 시도합니다.")

        save_qa_to_file(formatted_qa, file_name)
        print("QA 저장 완료")

