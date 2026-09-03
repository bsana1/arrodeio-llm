"""
Phase 3, step 3: score a (prompt, response) pair with the trained reward model.

    python scripts/score.py "Faça uma estrofe sobre a seca" "O sol racha a terra..."

The number is only meaningful *relative to other responses to the same prompt* —
the reward model was trained on which-is-better, not on an absolute scale.
Phase 4 (best-of-N) uses it to rank several generations.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RM_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "reward-model")


def load():
    if not os.path.isdir(RM_DIR):
        sys.exit("No reward model yet. Run:  python scripts/reward_model.py")
    tok = AutoTokenizer.from_pretrained(RM_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(RM_DIR)
    model.eval()
    return tok, model


@torch.no_grad()
def score(tok, model, prompt, response, max_len=256):
    enc = tok(prompt, response, truncation=True, max_length=max_len, return_tensors="pt")
    return model(**enc).logits.squeeze().item()


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python scripts/score.py "<prompt>" "<response>"')
    tok, model = load()
    print(f"{score(tok, model, sys.argv[1], sys.argv[2]):+.3f}")


if __name__ == "__main__":
    main()
