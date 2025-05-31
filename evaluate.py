# evaluate.py
"""
Compute basic QA metrics between predictions and gold answers.

Prediction JSONL  (from generate/answer.py):
{
  "question_id": "...",
  "answer": "The net income was $3.5 million."
}

Gold JSONL  (from prepare_dataset.py):
{
  "question_id": "...",
  "gold_answer": "3.5 million"
}

Outputs to stdout:
EM = 0.72
NumEM = 0.68
F1 = 0.79
"""

import argparse, json, re, string
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


# ---------- Normalisation helpers ---------- #
PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in PUNCT)
    text = " ".join(text.split())  # squeeze spaces
    return text


def get_numbers(text: str):
    return re.findall(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?", text.replace(",", ""))


# ---------- F1 helpers (SQuAD-style) ---------- #
def _token_f1(pred, truth):
    pred_toks = normalize(pred).split()
    truth_toks = normalize(truth).split()

    common = Counter(pred_toks) & Counter(truth_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(truth_toks)
    return 2 * precision * recall / (precision + recall)


# ---------- Main evaluation ---------- #
def evaluate(pred_file, gold_file):
    preds = {d["question_id"]: d["answer"] for d in load_jsonl(pred_file)}
    golds = {d["question_id"]: d["gold_answer"] for d in load_jsonl(gold_file)}

    em_total = num_total = f1_total = 0
    n = len(golds)

    for qid, gold_ans in golds.items():
        pred_ans = preds.get(qid, "")

        # EM
        em_total += int(normalize(pred_ans) == normalize(gold_ans))

        # Numeric exact
        nums_pred = get_numbers(pred_ans)
        nums_gold = get_numbers(gold_ans)
        num_total += int(nums_pred and nums_gold and nums_pred[0] == nums_gold[0])

        # F1
        f1_total += _token_f1(pred_ans, gold_ans)

    return {
        "EM": em_total / n,
        "NumEM": num_total / n,
        "F1": f1_total / n,
        "N": n,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True, help="Predictions JSONL")
    ap.add_argument("--gold_file", required=True, help="Gold JSONL")
    args = ap.parse_args()

    scores = evaluate(args.pred_file, args.gold_file)
    for k, v in scores.items():
        if k != "N":
            print(f"{k} = {v:.4f}")
    print(f"Total examples = {scores['N']}")
