from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")

def format_chain(response_schemas, qa_gen, json_format):
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

    return format_chain.invoke({"context": qa_gen, "json": json_format})