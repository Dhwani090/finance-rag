import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_corpus(corpus_dir):
    corpus = []
    ids = []
    for fname in os.listdir(corpus_dir):
        path = os.path.join(corpus_dir, fname)
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                text = ex.get("text", "")
                doc_id = ex.get("doc_id", "")
                corpus.append(text)
                ids.append(doc_id)
    return corpus, ids

def encode_texts(texts, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(embs)
    return np.vstack(embeddings)

def run_retrieval(query_file, corpus_dir, out_file, model_name="BAAI/bge-large-en-v1.5", top_k=10):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Loading corpus...")
    corpus_texts, corpus_ids = load_corpus(corpus_dir)
    
    print(f"Encoding corpus ({len(corpus_texts)} passages)...")
    corpus_embs = encode_texts(corpus_texts, model)
    
    print("Building FAISS index...")
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)    # cosine sim
    index.add(corpus_embs)
    
    print(f"Loading queries from {query_file}...")
    with open(query_file) as f:
        queries = [json.loads(l) for l in f]
    
    print("Encoding queries...")
    query_texts = [ex["expanded_question"] for ex in queries]
    query_embs = encode_texts(query_texts, model)
    
    print(f"Searching top-{top_k}...")
    D, I = index.search(query_embs, top_k)
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as out_f:
        for q_idx, ex in enumerate(queries):
            hits = []
            for rank, (score, doc_idx) in enumerate(zip(D[q_idx], I[q_idx])):
                hits.append({
                    "rank": rank + 1,
                    "score": float(score),
                    "doc_id": corpus_ids[doc_idx],
                    "text": corpus_texts[doc_idx]
                })
            out_record = {
                "question_id": ex["question_id"],
                "query": ex["expanded_question"],
                "hits": hits
            }
            json.dump(out_record, out_f)
            out_f.write("\n")
    
    print(f"Saved retrieval results → {out_file}")

if __name__ == "__main__":
    run_retrieval(
        query_file="dataset/FinQA/queries_expanded.jsonl",
        corpus_dir="dataset/FinQA/corpus",
        out_file="dataset/FinQA/queries_top10.jsonl",
        model_name="BAAI/bge-large-en-v1.5",
        top_k=10
    )
