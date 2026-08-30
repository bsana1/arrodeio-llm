"""
Talk to the Phase 2 SFT model (Tucano-1b1-Instruct + our cordel LoRA adapter).

    python scripts/chat.py

Type an instruction ("Glose o mote: «...»", "Faça uma estrofe sobre a cheia do
rio", ...). Ctrl-D or "sair" to quit. Slow on CPU — ~1 min per answer.
Pass --base to talk to the un-fine-tuned Tucano for comparison.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TucanoBR/Tucano-1b1-Instruct"
LORA_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "cordel-sft-lora")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="raw Tucano, no cordel adapter")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)

    if not args.base:
        if not os.path.isdir(LORA_DIR):
            sys.exit("No LoRA adapter yet. Run:  python scripts/sft.py")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()

    tag = "Tucano cru" if args.base else "arrodeio-llm (cordel)"
    print(f"{tag} — fale um mote ou um tema. ('sair' pra encerrar)\n")
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

        enc = tok.apply_chat_template([{"role": "user", "content": user}],
                                      tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt")
        print("  (…)", flush=True)
        out = model.generate(
            enc, attention_mask=torch.ones_like(enc), max_new_tokens=args.tokens,
            do_sample=True, temperature=args.temperature, top_k=50,
            repetition_penalty=1.2, pad_token_id=tok.eos_token_id,
        )
        print("\n" + tok.decode(out[0][enc.shape[1]:], skip_special_tokens=True).strip() + "\n")


if __name__ == "__main__":
    main()
