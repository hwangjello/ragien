import json
from evaluation import evaluate

# 파일 경로 정의
listed_file = "./result/listed_qao_hal_score.json"
csv_output = "./result/eval_result.csv"
json_output = "./result/eval_result.json"

# 데이터 로드
with open(listed_file, "r", encoding="utf-8") as f:
    data_json = json.load(f)

# 평가 실행
evaluate(
    data_json=data_json,
    csv_output=csv_output,
    json_output=json_output
)