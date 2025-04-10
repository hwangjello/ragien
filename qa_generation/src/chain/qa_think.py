from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")

def qa_think(qa_num, pc, example, context, requirements):
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
    return qa_think_chain.invoke({"num":qa_num,"prompt":pc,"example":example, "context": context, "user_req": requirements})
