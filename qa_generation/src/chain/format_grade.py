from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

deep = Ollama(model="deepseek-r1:14b")
llm = Ollama(model="gemma3:12b")

def format_grade(json_format, formatted_qa):
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

    return format_grader.invoke({"json": json_format, "answer": formatted_qa})