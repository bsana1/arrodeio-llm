"""
Inference for the fine-tuned cordel model (Phase 1b).

    python scripts/generate_ft.py "Vou contar uma história"
    python scripts/generate_ft.py "Quem com ferro fere" --tokens 200 --temperature 0.9
    python scripts/generate_ft.py "A donzela" --raw       # no few-shot priming
    python scripts/generate_ft.py "A donzela" --base       # raw pretrained model
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FT_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "cordel-ft")
BASE_MODEL = "pierreguillou/gpt2-small-portuguese"

# A couple of real stanzas as priming context. Even after fine-tuning, showing
# the model the target form up front sharply reduces prose drift.
FEWSHOT = """Eram doze cavaleiros
homens muito valorosos,
destemidos, animosos,
entre todos os guerreiros.

Vou contar uma história
de um matuto sertanejo
que trocou sua viola
por um sonho e um desejo.

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--raw", action="store_true", help="skip the few-shot priming stanzas")
    ap.add_argument("--base", action="store_true",
                    help="load the raw pretrained model instead of the fine-tuned one")
    args = ap.parse_args()

    src = BASE_MODEL if args.base else FT_DIR
    if not args.base and not os.path.isdir(FT_DIR):
        sys.exit("No fine-tuned model yet. Run:  python scripts/finetune.py")

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src)
    model.eval()

    context = args.prompt if args.raw else FEWSHOT + args.prompt
    ids = tok(context, return_tensors="pt").input_ids
    out = model.generate(
        ids, max_new_tokens=args.tokens, do_sample=True,
        temperature=args.temperature, top_k=args.top_k,
        repetition_penalty=1.4, pad_token_id=tok.eos_token_id,
    )
    generated = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(args.prompt + generated)


if __name__ == "__main__":
    main()
