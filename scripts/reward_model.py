"""
Phase 3, step 2: train the reward model.

Reads data/prefs.jsonl (run make_prefs.py first). Each row is a prompt with a
`chosen` and a `rejected` response. We train a scorer so that, for every pair,

    score(prompt, chosen)  >  score(prompt, rejected)

Loss is the Bradley-Terry / pairwise-logistic one:  -log σ(r_chosen - r_rejected).
The model never sees an absolute "quality score" — only which of two is better.

Backbone: BERTimbau (a Portuguese BERT). Small enough to train on a CPU in
~15 min. `AutoModelForSequenceClassification` with num_labels=1 = BERT + a
single linear head producing one number.

    python scripts/make_prefs.py
    python scripts/reward_model.py        # -> model/reward-model/
    python scripts/score.py "<prompt>" "<response>"
"""

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKBONE = "neuralmind/bert-base-portuguese-cased"
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data", "prefs.jsonl")
OUT_DIR = os.path.join(BASE, "..", "model", "reward-model")

# ---- config ---------------------------------------------------------
max_len = 256
batch_size = 8
learning_rate = 2e-5
epochs = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------------

torch.manual_seed(1337)
if device == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)


def score(model, tok, prompts, responses):
    enc = tok(prompts, responses, truncation=True, max_length=max_len,
              padding=True, return_tensors="pt").to(device)
    return model(**enc).logits.squeeze(-1)      # (B,)


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    torch.manual_seed(0)
    perm = torch.randperm(len(rows)).tolist()
    rows = [rows[i] for i in perm]
    n_val = max(64, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    print(f"{len(train)} train / {len(val)} val preference pairs")

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = 0
        margins = []
        for i in range(0, len(val), batch_size):
            b = val[i:i + batch_size]
            p = [x["prompt"] for x in b]
            rc = score(model, tok, p, [x["chosen"] for x in b])
            rr = score(model, tok, p, [x["rejected"] for x in b])
            correct += (rc > rr).sum().item()
            margins += (rc - rr).tolist()
        model.train()
        return correct / len(val), sum(margins) / len(margins)

    acc, margin = evaluate()
    print(f"step 0 (untrained): val acc {acc:.3f}, mean margin {margin:+.3f}")

    step, t0 = 0, time.time()
    for ep in range(epochs):
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), batch_size):
            b = [train[j] for j in order[i:i + batch_size]]
            p = [x["prompt"] for x in b]
            rc = score(model, tok, p, [x["chosen"] for x in b])
            rr = score(model, tok, p, [x["rejected"] for x in b])
            loss = -F.logsigmoid(rc - rr).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 60 == 0:
                acc, margin = evaluate()
                print(f"ep {ep} step {step}  loss {loss.item():.4f}  "
                      f"val acc {acc:.3f}  margin {margin:+.3f}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    acc, margin = evaluate()
    print(f"\nfinal: val acc {acc:.3f}, mean margin {margin:+.3f}")
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
