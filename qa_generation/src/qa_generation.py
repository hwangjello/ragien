from chain.qa_think import qa_think
from chain.answer_grade import answer_grade
from chain.format_chain import format_chain
from chain.qa_one_think import qa_one_think
from chain.pc_one_think import pc_one_think
from chain.pc_think import pc_think
from chain.format_grade import format_grade
from chain.utils import generate_json_format, generate_response_schemas, make_noun, nonOri, hal_score, is_valid_qa, save_qa_to_file

import json
import re

def qa_generation(papers, file_name, example=" ", requirements=" ", qa_num=5, threshold=0.8):
    for sample in datas:
        context = json.dumps(sample, ensure_ascii=False, indent=4)
        while True:
            try:
                pc = pc_think(requirements,qa_num)
                qa_think_out = qa_think(num=qa_num, prompt=pc, example=example, context=context, user_req=requirements)
                qa_gen = re.sub(r'<think>.*?</think>\n*', '', qa_think_out, flags=re.DOTALL)
                
                json_format = generate_json_format(qa_num)
                qa_response_schemas = generate_response_schemas(qa_num)

                formatted_qa = format_chain(qa_response_schemas, qa_gen,json_format)
                qa_grade = answer_grade(context,formatted_qa)
                form_grade = format_grade(json_format,formatted_qa)
                is_valid = is_valid_qa(formatted_qa)
                scores = hal_score(formatted_qa, context, qa_num)
                print(scores)

                if not all(score >= threshold for score in scores):
                    continue
                if qa_grade["binary_score"] == "yes" and form_grade["binary_score"] == "yes" and is_valid:
                    break
                else:
                    print("Non an adequate QA. QA re-generation.")
                    continue

            except Exception as e:
                print(f"Error Occured: {e}. retry.")

        save_qa_to_file(formatted_qa, file_name)
        print("QA saved.")