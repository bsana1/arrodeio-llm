"""
Build the tokenizer that train.py and generate.py will use.

Reads data/corpus.txt, learns a byte-level BPE vocab, writes model/tokenizer.json.
Run this after prepare_data.py and before train.py.

Usage:
    python scripts/train_tokenizer.py                 # BPE, 768 merges (vocab 1024)
    python scripts/train_tokenizer.py --merges 1200
    python scripts/train_tokenizer.py --kind char     # the original char-level setup
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import tokenizer as tk

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
OUT = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["bpe", "char"], default="bpe")
    ap.add_argument("--merges", type=int, default=768,
                    help="BPE only: number of merge rules to learn (vocab = 256 + this)")
    args = ap.parse_args()

    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()

    if args.kind == "char":
        t = tk.CharTokenizer.train(text)
    else:
        print(f"Training BPE ({args.merges} merges) on {len(text):,} chars "
              f"— pure Python, give it a minute...")
        t = tk.BPETokenizer.train(text, n_merges=args.merges)

    tk.save(t, OUT)

    ids = t.encode(text)
    print(f"\nTokenizer: {t.kind}, vocab size {t.vocab_size}")
    print(f"Corpus: {len(text):,} chars -> {len(ids):,} tokens "
          f"({len(text) / max(len(ids), 1):.2f} chars/token)")

    demo = "Quem com ferro fere, com ferro será ferido."
    pieces = [t.decode([i]) for i in t.encode(demo)]
    print(f"\n  {demo!r}")
    print(f"  -> {pieces}")
    print(f"\nWrote {os.path.relpath(OUT)} — next: python scripts/train.py")


if __name__ == "__main__":
    main()
