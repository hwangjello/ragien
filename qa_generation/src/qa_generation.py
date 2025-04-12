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


def qa_generation(datas, file_name, example=" ", requirements=" ", qa_num=5, threshold=0.8):
    retry_count = 0  # Ollama 오류로 인해 처음부터 다시 시작하는 횟수 카운터
    pc = pc_think(requirements, qa_num)
    while True:
        try:
            for sample in datas:
                context = json.dumps(sample, ensure_ascii=False, indent=4)
                while True:
                    try:
                        qa_think_out = qa_think(qa_num, pc, example, context, requirements)
                        qa_gen = re.sub(r'<think>.*?</think>\n*', '', qa_think_out, flags=re.DOTALL)
                        
                        json_format = generate_json_format(qa_num)
                        qa_response_schemas = generate_response_schemas(qa_num)

                        formatted_qa = format_chain(qa_response_schemas, qa_gen, json_format)
                        qa_grade = answer_grade(context, formatted_qa)
                        form_grade = format_grade(json_format, formatted_qa)
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
                        err_msg = str(e)
                        if "llama runner" in err_msg or "Ollama call failed" in err_msg:
                            print(f"Ollama error detected. Restarting from beginning. Details: {err_msg}")
                            raise RuntimeError("Restarting due to Ollama error")
                        else:
                            print(f"Error Occurred: {e}. retry.")
                            continue

                # suffix가 붙은 파일 이름 구성
                full_file_name = f"{file_name}_{retry_count}.json"
                save_qa_to_file(formatted_qa, full_file_name)
                print(f"QA saved.")

            break  # 모든 데이터 성공 처리 완료 → while 탈출

        except RuntimeError:
            retry_count += 1
            print(f"Restarting QA generation process. Retry count: {retry_count}")
            continue  # 다시 처음부터 시작