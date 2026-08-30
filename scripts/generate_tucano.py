"""
Talk to the raw Tucano-1b1-Instruct (Brazilian Portuguese, instruction-tuned).
This is the *un-fine-tuned* baseline — the starting point for Phase 2.

    python scripts/generate_tucano.py "Faça uma estrofe de cordel sobre a seca"
    python scripts/generate_tucano.py "Glose o mote: «Água mole em pedra dura...»" --tokens 150

Slow on CPU (~1.5 tok/s in bf16). First run downloads ~2.2 GB.
"""

import argparse
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TucanoBR/Tucano-1b1-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()

    enc = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    attn = torch.ones_like(enc)
    print("(gerando… ~1.5 tok/s)\n", flush=True)
    out = model.generate(
        enc, attention_mask=attn, max_new_tokens=args.tokens, do_sample=True,
        temperature=args.temperature, top_k=args.top_k, top_p=1.0,
        repetition_penalty=1.2, pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0][enc.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
