import os, json
import openai
import time
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

openai.api_key = api_key

def expand_query(q, max_retries=3, delay=1):
    prompt = "Extract keywords:\n{}".format(q)
    for attempt in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"\nRate limit hit, waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
            print(f"\nError in expand_query: {str(e)}")
            return q  # Return original query if expansion fails
    return q

def run(in_file, out_file):
    directory = os.path.dirname(out_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Load existing progress if any
    processed_ids = set()
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    processed_ids.add(ex.get('_id', ''))
                except:
                    continue
    
    # Load and process queries
    with open(in_file) as f:
        lines = [json.loads(l) for l in f]
    
    with open(out_file, "a") as out:  # Append mode
        for ex in tqdm(lines):
            if ex.get('_id', '') in processed_ids:
                continue
                
            ex["expanded_question"] = expand_query(ex["text"])
            json.dump(ex, out)
            out.write("\n")
            out.flush()  # Ensure data is written immediately
            time.sleep(0.5)  # Add delay between API calls

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="dataset/finqa/queries.jsonl")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()
    run(args.in_file, args.out_file)
