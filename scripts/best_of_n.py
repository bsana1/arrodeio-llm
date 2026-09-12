"""
Phase 4: best-of-N reranking.

No training. For one instruction:
  1. generate N candidate stanzas from the Phase 2 model
  2. score each with the Phase 3 reward model
  3. return the highest-scoring one

This is the cheapest way a reward model improves output — you spend N times the
compute at inference and keep the best sample. (It's also the seed for
"rejection-sampling fine-tuning": collect the best-of-N picks and SFT on them.)

    python scripts/best_of_n.py "Glose o mote: «O apressado come cru.»" -n 5
    python scripts/best_of_n.py "Faça uma estrofe sobre a lua" -n 6 --model gpt2-ft

Tucano is slow on CPU (~1 min per candidate). --model gpt2-ft is much faster for
trying the mechanism out.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch

BASE = os.path.dirname(__file__)
sys.path.insert(0, BASE)
from score import load as load_rm, score as rm_score

TUCANO = "TucanoBR/Tucano-1b1-Instruct"
LORA = os.path.join(BASE, "..", "model", "cordel-sft-lora")
GPT2_FT = os.path.join(BASE, "..", "model", "cordel-ft")

FEWSHOT = (
    "Eram doze cavaleiros\nhomens muito valorosos,\ndestemidos, animosos,\n"
    "entre todos os guerreiros.\n\n"
)


def gen_tucano(prompt, n, max_new_tokens, temperature):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(TUCANO)
    model = AutoModelForCausalLM.from_pretrained(TUCANO, dtype=torch.bfloat16)
    if os.path.isdir(LORA):
        model = PeftModel.from_pretrained(model, LORA)
    model.eval()
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    outs = []
    for _ in range(n):
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=max_new_tokens, do_sample=True,
                           temperature=temperature, top_k=50, repetition_penalty=1.2,
                           pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def gen_gpt2_ft(prompt, n, max_new_tokens, temperature):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GPT2_FT)
    model = AutoModelForCausalLM.from_pretrained(GPT2_FT)
    model.eval()
    theme = prompt.split("«")[-1].rstrip("».") if "«" in prompt else prompt
    ids = tok(FEWSHOT + theme + "\n", return_tensors="pt").input_ids
    outs = []
    for _ in range(n):
        g = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                           temperature=temperature, top_k=40, repetition_penalty=1.4,
                           pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--model", choices=["tucano", "gpt2-ft"], default="tucano")
    ap.add_argument("--tokens", type=int, default=110)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    print(f"generating {args.n} candidates ({args.model}) ...", flush=True)
    gen = gen_tucano if args.model == "tucano" else gen_gpt2_ft
    cands = gen(args.prompt, args.n, args.tokens, args.temperature)

    tok, rm = load_rm()
    scored = sorted(((rm_score(tok, rm, args.prompt, c), c) for c in cands),
                    key=lambda x: -x[0])

    for rank, (s, c) in enumerate(scored, 1):
        mark = "  ← best" if rank == 1 else ""
        print(f"\n[{rank}] reward {s:+.3f}{mark}\n" + c)

    print("\n" + "=" * 50)
    print("BEST OF N:\n" + scored[0][1])


if __name__ == "__main__":
    main()
