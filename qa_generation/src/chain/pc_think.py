from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")


def pc_think(requirements, qa_num):
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

    return pc_think_chain.invoke({"user_req": requirements,"num":qa_num})