# rerank.py

import os, json, argparse
from tqdm import tqdm
import torch
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            json.dump(r, f)
            f.write("\n")


def rerank_file(
    in_file: str,
    out_file: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
    top_k: int | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading cross-encoder {model_name} on {device} …")
    model = CrossEncoder(model_name, device=device)

    data = load_jsonl(in_file)
    print(f"Scoring {sum(len(d['hits']) for d in data):,} (query, doc) pairs")

    for record in tqdm(data):
        query = record["query"]
        pairs = [(query, hit["text"]) for hit in record["hits"]]
        scores = model.predict(pairs, batch_size=batch_size).tolist()

        for hit, s in zip(record["hits"], scores):
            hit["score"] = float(s)
        record["hits"].sort(key=lambda h: h["score"], reverse=True)

        if top_k:
            record["hits"] = record["hits"][: top_k]

        for r, hit in enumerate(record["hits"], 1):
            hit["rank"] = r

    save_jsonl(data, out_file)
    print(f"Saved reranked results → {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    rerank_file(
        in_file=args.in_file,
        out_file=args.out_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        top_k=args.top_k,
    )
