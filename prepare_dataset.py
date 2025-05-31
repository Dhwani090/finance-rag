# prepare_dataset.py

import os
import json
from datasets import load_dataset

def prepare_finqa(out_dir="dataset/finqa"):
    os.makedirs(out_dir, exist_ok=True)
    
    print("Downloading FinQA...")
    ds = load_dataset("finqa")   # Using the correct dataset name
    
    print("Processing...")
    for split in ["queries", "validation", "test"]:
        examples = ds[split]
        out_file = os.path.join(out_dir, "{}.jsonl".format(split))
        
        with open(out_file, "w") as f:
            for ex in examples:
                json.dump({
                    "question_id": ex.get("id", ""),
                    "question": ex["question"],
                    "gold_answer": ex.get("answer", ""),
                    "table": ex.get("table_text", ""),
                    "context": ex.get("passage_text", "")
                }, f)
                f.write("\n")
        
        print("Wrote {}".format(out_file))

if __name__ == "__main__":
    prepare_finqa()
