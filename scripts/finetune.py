"""
Phase 1b: style fine-tuning.

Takes a pretrained Portuguese GPT-2 (which already knows the language) and
nudges it toward cordel / ditado voice by continuing to train it on
data/corpus.txt — but only the top few transformer layers, so it keeps its
grip on Portuguese and just picks up the rhythm, vocabulary and rhyme.

This is *continued pretraining* (plain next-token prediction on the corpus),
not instruction tuning — that's Phase 2 (scripts/sft.py, planned).

Usage:
    python scripts/prepare_data.py        # make sure data/corpus.txt is current
    python scripts/finetune.py

Then:
    python scripts/generate_ft.py "Vou contar uma história"
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "pierreguillou/gpt2-small-portuguese"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "cordel-ft")

# ---- config -----------------------------------------------------------
block_size = 192          # tokens of context per training window
batch_size = 4            # small: we're on CPU
unfreeze_top_n = 6        # train the top half of the 12 transformer blocks
learning_rate = 1e-4      # push harder — the encyclopedia prior is stubborn
max_iters = 450           # sweet spot: style has landed, not yet memorising
eval_interval = 75
save_interval = 150       # keep a rolling checkpoint from here on
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

torch.manual_seed(1337)


def main():
    print(f"Device: {device}")
    print(f"Loading {BASE_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    # Freeze everything, then thaw the top few blocks + the final layernorm.
    for p in model.parameters():
        p.requires_grad = False
    trainable_mods = list(model.transformer.h[-unfreeze_top_n:]) + [model.transformer.ln_f]
    for m in trainable_mods:
        for p in m.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Training {n_train:,} / {n_total:,} params "
          f"(top {unfreeze_top_n} blocks + ln_f)")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text).input_ids, dtype=torch.long)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"Corpus: {len(text):,} chars -> {len(ids):,} tokens "
          f"(train {len(train_ids):,} / val {len(val_ids):,})")

    def get_batch(split):
        d = train_ids if split == "train" else val_ids
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix]).to(device)
        return x

    @torch.no_grad()
    def eval_loss(iters=20):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(iters)
            for k in range(iters):
                x = get_batch(split)
                losses[k] = model(input_ids=x, labels=x).loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    @torch.no_grad()
    def sample(prompt="Vou contar uma história", new_tokens=60):
        model.eval()
        x = tok(prompt, return_tensors="pt").input_ids.to(device)
        y = model.generate(x, max_new_tokens=new_tokens, do_sample=True,
                           temperature=0.8, top_k=40, repetition_penalty=1.3,
                           pad_token_id=tok.eos_token_id)
        model.train()
        return tok.decode(y[0], skip_special_tokens=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    def save():
        model.save_pretrained(OUT_DIR)
        tok.save_pretrained(OUT_DIR)

    # NOTE: unlike pretraining (train.py), we do NOT keep the lowest-val-loss
    # checkpoint here. For style transfer, val loss bottoms out almost
    # immediately (~step 75) while the *style* keeps improving for hundreds
    # more steps. So we just let it run to max_iters and keep a rolling save
    # from save_interval on — max_iters is tuned to stop before it memorises.
    model.train()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            L = eval_loss()
            print(f"step {it}: train {L['train']:.4f}, val {L['val']:.4f}")
            print(f"  sample: {sample()!r}")
        if it >= save_interval and (it % eval_interval == 0 or it == max_iters - 1):
            save()

        x = get_batch("train")
        loss = model(input_ids=x, labels=x).loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    save()
    print(f"\nFine-tuned model saved to {OUT_DIR}")
    print("Try: python scripts/generate_ft.py \"Vou contar uma história\"")


if __name__ == "__main__":
    main()
