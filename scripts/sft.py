"""
Phase 2: Supervised fine-tuning (SFT).

Takes the Phase 1b style model (model/cordel-ft/) and teaches it to *respond to
an instruction* — "glose este mote", "faça uma estrofe sobre X" — instead of
just continuing text. Trains on data/sft.jsonl: {"prompt": ..., "response": ...}.

The one idea that makes this SFT and not just more pretraining: the loss is
**masked on the prompt tokens** (labels = -100 there), so the model is only
graded on producing the response.

Usage:
    python scripts/sft.py
    python scripts/chat.py            # then talk to it
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(__file__)
IN_MODEL = os.path.join(BASE_DIR, "..", "model", "cordel-ft")      # Phase 1b output
OUT_DIR = os.path.join(BASE_DIR, "..", "model", "cordel-sft")
DATA_PATHS = [
    os.path.join(BASE_DIR, "..", "data", "sft.jsonl"),        # hand-written mote->glosa
    os.path.join(BASE_DIR, "..", "data", "sft_synth.jsonl"),  # backtranslated (run make_sft_data.py)
]

# The template. Keep it identical in chat.py.
PROMPT_TMPL = "{prompt}\n\n"

# ---- config -----------------------------------------------------------
batch_size = 4
unfreeze_top_n = 4       # ~600 examples now, can afford a bit more
learning_rate = 3e-5
epochs = 3
max_steps = 700          # cap CPU time regardless of dataset size
eval_interval = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

torch.manual_seed(1337)


def main():
    if not os.path.isdir(IN_MODEL):
        sys.exit("Need the Phase 1b model first. Run:  python scripts/finetune.py")

    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(IN_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(IN_MODEL).to(device)

    for p in model.parameters():
        p.requires_grad = False
    for m in list(model.transformer.h[-unfreeze_top_n:]) + [model.transformer.ln_f]:
        for p in m.parameters():
            p.requires_grad = True

    # ---- build the training examples ----
    pairs = []
    for path in DATA_PATHS:
        if not os.path.exists(path):
            print(f"  (skipping {os.path.basename(path)} — not found)")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    if not pairs:
        sys.exit("No SFT data. Write data/sft.jsonl and/or run scripts/make_sft_data.py")
    print(f"{len(pairs)} SFT pairs, up to {epochs} epochs / {max_steps} steps")

    examples = []
    for ex in pairs:
        p_ids = tok(PROMPT_TMPL.format(prompt=ex["prompt"])).input_ids
        r_ids = tok(ex["response"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        input_ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids        # <-- prompt is masked out
        examples.append((input_ids, labels))

    def get_batch(step):
        # simple shuffled-ish sampling
        idx = torch.randint(len(examples), (batch_size,))
        rows = [examples[i] for i in idx]
        maxlen = max(len(r[0]) for r in rows)
        x = torch.full((batch_size, maxlen), tok.pad_token_id, dtype=torch.long)
        y = torch.full((batch_size, maxlen), -100, dtype=torch.long)
        mask = torch.zeros((batch_size, maxlen), dtype=torch.long)
        for i, (ii, ll) in enumerate(rows):
            x[i, : len(ii)] = torch.tensor(ii)
            y[i, : len(ll)] = torch.tensor(ll)
            mask[i, : len(ii)] = 1
        return x.to(device), y.to(device), mask.to(device)

    @torch.no_grad()
    def demo():
        model.eval()
        for prompt in ["Glose o mote: «Gato escaldado tem medo de água fria.»",
                       "Faça uma estrofe de cordel sobre a lua no sertão."]:
            enc = tok(PROMPT_TMPL.format(prompt=prompt), return_tensors="pt").to(device)
            out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=80, do_sample=True, temperature=0.8,
                                 top_k=40, repetition_penalty=1.3,
                                 pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            print(f"  › {prompt}")
            print("   " + tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).replace("\n", "\n   "))
        model.train()

    steps = min(max_steps, max(1, (len(examples) * epochs) // batch_size))
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    model.train()
    for step in range(steps):
        if step % eval_interval == 0 or step == steps - 1:
            x, y, m = get_batch(step)
            with torch.no_grad():
                vloss = model(input_ids=x, attention_mask=m, labels=y).loss.item()
            print(f"step {step}/{steps}: loss {vloss:.4f}")
            demo()
        x, y, m = get_batch(step)
        loss = model(input_ids=x, attention_mask=m, labels=y).loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"\nSaved to {OUT_DIR}. Try:  python scripts/chat.py")


if __name__ == "__main__":
    main()
