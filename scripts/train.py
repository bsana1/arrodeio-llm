"""
Phase 1: Pretraining.

Trains TinyGPT from scratch on data/corpus.txt (run prepare_data.py first).

Usage:
    python scripts/train.py
"""

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.gpt import TinyGPT
from model import tokenizer as tokenizer_mod

# ---- config -----------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "checkpoint.pt")

block_size = 192
batch_size = 32
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.15
learning_rate = 3e-4
max_iters = 4000
eval_interval = 500
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------------------

torch.manual_seed(1337)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

if not os.path.exists(TOKENIZER_PATH):
    sys.exit("No tokenizer found. Run:  python scripts/train_tokenizer.py")

tok = tokenizer_mod.load(TOKENIZER_PATH)
vocab_size = tok.vocab_size


def encode(s):
    return tok.encode(s)


def decode(ids):
    return tok.decode(ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(model, prompt, max_new_tokens=200, temperature=0.8):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature)
    model.train()
    return decode(out[0].tolist())


def main():
    print(f"Device: {device}")
    print(f"Tokenizer: {tok.kind}, vocab size {vocab_size}")
    print(f"Corpus: {len(text):,} chars -> {len(data):,} tokens "
          f"({block_size} tokens = {len(text) / max(len(data), 1) * block_size:.0f} chars of context)")

    model = TinyGPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    cfg = dict(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd,
               n_head=n_head, n_layer=n_layer, dropout=dropout)

    def save(step):
        torch.save(
            {"model_state": model.state_dict(), "tokenizer": tok.state(),
             "config": cfg, "step": step},
            CHECKPOINT_PATH,
        )

    # On a corpus this small the model starts memorising well before max_iters,
    # so we keep the checkpoint from the *lowest val loss*, not the last step.
    best_val = float("inf")
    best_step = 0

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            marker = ""
            if losses["val"] < best_val:
                best_val, best_step = losses["val"], it
                save(it)
                marker = "  <- saved (best val)"
            print(f"step {it}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}{marker}")
            # Show a live sample so you can literally watch quality improve
            seed = "Quem" if "Quem" in text[:5000] else text[:4]
            print(f"  sample: {sample(model, seed, max_new_tokens=80)!r}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\nBest val loss {best_val:.4f} at step {best_step} — "
          f"that checkpoint is saved at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
