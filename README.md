# FinanceRAG

A comprehensive Retrieval-Augmented Generation (RAG) system for financial question answering.

## Overview

This project implements a full pipeline for financial question answering using RAG:
- Query expansion and preprocessing
- Document retrieval
- Reranking
- Answer generation

## Project Structure

```
finance-rag/
├── finance-rag/              # Core package
│   ├── generate/            # Answer generation modules
│   │   └── answer.py       # LLM-based answer generation
│   └── retrieval/          # Document retrieval modules
│       └── retrieve.py     # Vector-based retrieval
├── dataset/                # Dataset directory
│   └── finqa/             # FinQA dataset files
├── pre_retrieval.py       # Query preprocessing and expansion
├── prepare_dataset.py     # Dataset preparation utilities
├── rerank.py             # Document reranking
├── run.sh               # Main execution script
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
- Create a `.env` file with your API keys
- Required keys: `OPENAI_API_KEY`

## Usage

Run the complete pipeline:
```bash
./run.sh
```

Or run individual components:
```bash
python pre_retrieval.py --out_file dataset/finqa/output.jsonl
python rerank.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.