#!/usr/bin/env bash
# run.sh
#
# One-stop script to run the FinanceRAG pipeline for a single dataset split.
# -------------------------------------------------------------------------
# Usage:
#   ./run.sh finqa train 200 10
#          ^dataset ^split ^retriever_k ^rerank_k
#
# Example (default values):
#   ./run.sh finqa train
#
# Requirements:
#   * OPENAI_API_KEY  in your environment  (export or `.env`)
#   * The dataset must already be prepared by prepare_dataset.py

set -euo pipefail

# ----------------- Parse CLI ---------------- #
DATASET=${1:-finqa}          # folder name under dataset/
SPLIT=${2:-queries}            # queries / validation / test
TOP_K_RETR=${3:-200}         # first-pass dense retriever depth
TOP_K_RERANK=${4:-10}        # cross-encoder depth

DS_DIR="dataset/${DATASET}"
PROCESSED_DIR="${DS_DIR}/processed"
RESULT_DIR="${DS_DIR}/results/${SPLIT}"
mkdir -p "$RESULT_DIR"

RAW_Q="${DS_DIR}/${SPLIT}.jsonl"
EXP_Q="${PROCESSED_DIR}/${SPLIT}_exp.jsonl"
TOPK_JSON="${RESULT_DIR}/top${TOP_K_RETR}.jsonl"
RERANK_JSON="${RESULT_DIR}/top${TOP_K_RERANK}_reranked.jsonl"
ANS_JSON="${RESULT_DIR}/answers.jsonl"
METRICS_TXT="${RESULT_DIR}/metrics.txt"

echo "=== [1/5] Query-expansion ==="
python3 pre_retrieval.py  --in_file "$RAW_Q"   --out_file "$EXP_Q"
...
 \
  --in_file "$RAW_Q" \
  --out_file "$EXP_Q"

echo "=== [2/5] Dense retrieval (top $TOP_K_RETR) ==="
python financerag/retrieval/retrieve.py \
  --query_file "$EXP_Q" \
  --corpus_dir "${DS_DIR}/corpus" \
  --out_file "$TOPK_JSON" \
  --top_k "$TOP_K_RETR"

echo "=== [3/5] Rerank (top $TOP_K_RERANK) ==="
python rerank.py \
  --in_file "$TOPK_JSON" \
  --out_file "$RERANK_JSON" \
  --top_k "$TOP_K_RERANK"

echo "=== [4/5] LLM generation ==="
python generate/answer.py \
  --in_file "$RERANK_JSON" \
  --out_file "$ANS_JSON" \
  --model gpt-4o

echo "=== [5/5] Evaluation ==="
python evaluate.py \
  --pred_file "$ANS_JSON" \
  --gold_file "$RAW_Q" | tee "$METRICS_TXT"

echo "Pipeline complete! Metrics saved to $METRICS_TXT"
