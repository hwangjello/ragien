# evaluation.py
from metrics.bleu import bleu_n
from metrics.rouge import rouge_l
from metrics.wmd import wmd_score
from metrics.embedding import get_bert_score, get_sbert_score
from metrics.hallucination import compute_hal_score
from utils.preprocessing import filter_artwork_by_id
import pandas as pd

def evaluate(data_json, csv_output, json_output):
    results = []

    for artwork_id, content in data_json.items():
        questions = content["question"]
        ground_truths = content["ground_truth"]
        answers = content["answer"]
        source = filter_artwork_by_id(artwork_id[5:])

        for idx, (question, ground_truth, answer) in enumerate(zip(questions, ground_truths, answers)):
            sentences = [ground_truth, answer]

            bleu = bleu_n(ground_truth, answer)
            rouge = rouge_l(ground_truth, answer)
            wmd = wmd_score(ground_truth, answer)
            bert = get_bert_score(sentences)
            sbert = get_sbert_score(sentences)
            hal_score = compute_hal_score(ground_truth, answer, source)

            if sbert > 0 and hal_score > 0:
                fin_score = 2.0 / ((1.0 / sbert) + (1.0 / hal_score))
            else:
                fin_score = 0.0

            results.append([
                artwork_id, idx + 1, question, answer, ground_truth,
                bleu, rouge, wmd, bert, sbert, hal_score, fin_score
            ])

            print(f"{artwork_id} - {idx+1}/{len(questions)} sbert score: {sbert:.4f}, hallucination score: {hal_score:.4f} -> final score: {fin_score:.4f}")

    df_results = pd.DataFrame(results, columns=[
        "paper_id", "Question_Number", "Question", "Answer", "Ground Truth",
        "BLEU", "ROUGE-L", "WMD", "BERT", "SBERT", "Hal_score", "Tot_score"
    ])
    df_results.to_csv(csv_output, index=False, encoding="utf-8")
    print(f"Evaluation Completed! The result saved in CSV file named '{csv_output}'.")

    df_results.to_json(json_output, orient='records', force_ascii=False, indent=4)
    print(f"Evaluation Completed! The result saved in JSON file named '{json_output}'.")

