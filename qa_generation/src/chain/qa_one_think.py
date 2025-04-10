from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")

def qa_one_think(pc_one, context, requirements):
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
    return qa_think_one_chain.invoke({"prompt":pc_one, "context": context, "user_req": requirements})