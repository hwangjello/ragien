from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")

def answer_grade(context, formatted_qa):
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

    return answer_grader.invoke({"context": context,"answer": formatted_qa})