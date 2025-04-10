from chain.qa_one_think import qa_one_think
from chain.pc_one_think import pc_one_think
import json
import re
import random

def ex_generation(datas, requirements=" "):
    sample = random.choice(datas)
    context = json.dumps(sample, ensure_ascii=False, indent=4)

    pc_one = pc_one_think(requirements)
    qa_think_one = qa_one_think(pc_one, context, requirements)
    ex_qa = qa_think_one
    return ex_qa