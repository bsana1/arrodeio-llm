"""
Phase 3: Inference.

Load a trained checkpoint and generate text from a prompt.

Usage:
    python scripts/generate.py "Quem"
    python scripts/generate.py "Vou contar uma historia" --temperature 1.0 --tokens 300
"""

import argparse
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.gpt import TinyGPT
from model import tokenizer as tokenizer_mod

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "checkpoint.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Seed text to continue from")
    parser.add_argument("--tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=None, help="Optional top-k sampling cutoff")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    tok = tokenizer_mod.from_state(checkpoint["tokenizer"])
    model = TinyGPT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ids = tok.encode(args.prompt)
    if not ids:
        raise ValueError("Prompt is empty after tokenization — give it some text.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
