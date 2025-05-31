# generate/answer.py

import os, json, argparse, tiktoken
from tqdm import tqdm
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

ENC = tiktoken.encoding_for_model("gpt-4o")  # tokeniser


def num_tokens(text: str) -> int:
    return len(ENC.encode(text))


def build_ctx(hits, limit_tokens):
    """Concatenate docs until token budget reached."""
    ctx, total = [], 0
    for h in hits:
        t = h["text"].strip()
        tks = num_tokens(t)
        if total + tks > limit_tokens:
            break
        ctx.append(t)
        total += tks
    return "\n\n".join(ctx)


def call_llm(model, system_msg, user_msg, max_tokens=256):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"].strip()


def choose_answer(ans1, ans2):
    import re

    nums1 = set(re.findall(r"\d[\d,\.]*", ans1))
    nums2 = set(re.findall(r"\d[\d,\.]*", ans2))
    common = nums1 & nums2
    if common:
        return ans1 if len(ans1) <= len(ans2) else ans2
    return ans1  # fallback


def generate(
    in_file: str,
    out_file: str,
    model: str = "gpt-4o",
    max_ctx_tokens: int = 32000,
    answer_max_tokens: int = 256,
):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(in_file) as f:
        data = [json.loads(l) for l in f]

    results = []
    for record in tqdm(data):
        qid = record["question_id"]
        question = record["query"]
        hits = record["hits"]

        system_msg = "You are a helpful financial QA assistant."
        ctx = build_ctx(hits, max_ctx_tokens - 1024)  # leave room

        full_prompt = f"Question:\n{question}\n\nContext:\n{ctx}\n\nAnswer the question."

        if num_tokens(full_prompt) <= max_ctx_tokens:
            a1 = call_llm(model, system_msg, full_prompt, answer_max_tokens)
            final = a1
            raw = [a1]
        else:
            # split context in half
            mid = len(hits) // 2
            ctx1 = build_ctx(hits[:mid], max_ctx_tokens - 1024)
            ctx2 = build_ctx(hits[mid:], max_ctx_tokens - 1024)

            p1 = f"Question:\n{question}\n\nContext:\n{ctx1}\n\nAnswer the question."
            p2 = f"Question:\n{question}\n\nContext:\n{ctx2}\n\nAnswer the question."

            a1 = call_llm(model, system_msg, p1, answer_max_tokens)
            a2 = call_llm(model, system_msg, p2, answer_max_tokens)
            final = choose_answer(a1, a2)
            raw = [a1, a2]

        results.append(
            {
                "question_id": qid,
                "answer": final,
                "raw_answers": raw,
                "model": model,
            }
        )

    with open(out_file, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")
    print(f"Saved answers → {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max_ctx_tokens", type=int, default=32000)
    ap.add_argument("--answer_max_tokens", type=int, default=256)
    args = ap.parse_args()

    generate(
        in_file=args.in_file,
        out_file=args.out_file,
        model=args.model,
        max_ctx_tokens=args.max_ctx_tokens,
        answer_max_tokens=args.answer_max_tokens,
    )
