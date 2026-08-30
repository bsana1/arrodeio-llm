"""
Talk to the SFT model (Phase 2).

    python scripts/chat.py

Type an instruction ("Glose o mote: «...»", "Faça uma estrofe sobre a cheia do
rio", ...) and it answers in cordel voice. Ctrl-D or "sair" to quit.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "cordel-sft")
PROMPT_TMPL = "{prompt}\n\n"          # must match sft.py

TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 120


def main():
    if not os.path.isdir(SFT_DIR):
        sys.exit("No SFT model yet. Run:  python scripts/sft.py")

    tok = AutoTokenizer.from_pretrained(SFT_DIR)
    model = AutoModelForCausalLM.from_pretrained(SFT_DIR)
    model.eval()

    print("arrodeio-llm — fale um mote ou um tema. ('sair' pra encerrar)\n")
    while True:
        try:
            user = input("você> ").strip()
        except EOFError:
            print()
            break
        if user.lower() in {"sair", "quit", "exit"}:
            break
        if not user:
            continue

        ids = tok(PROMPT_TMPL.format(prompt=user), return_tensors="pt").input_ids
        out = model.generate(
            ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            temperature=TEMPERATURE, top_k=TOP_K, repetition_penalty=1.3,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
        reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n{reply}\n")


if __name__ == "__main__":
    main()
