import json

def json_loader(data_name):
    with open(data_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 항상 리스트 형태로 반환
    if isinstance(data, list):
        return data
    else:
        return [data]