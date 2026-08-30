"""
Phase 2: Supervised fine-tuning (SFT) with LoRA.

TucanoBR/Tucano-1b1-Instruct already follows instructions and speaks Brazilian
Portuguese. This teaches it to answer in *cordel* form and cut the chatty
preamble, training on:
  - data/sft.jsonl        — hand-written  mote -> glosa  pairs
  - data/sft_synth.jsonl  — instruction-backtranslated from the real corpus
                            (run scripts/make_sft_data.py first)

Two ideas worth understanding:

  LoRA — the 1.1B base is frozen. We bolt small low-rank matrices onto the
  attention projections and train only those (~0.4% of the weights). The saved
  adapter is ~9 MB.

  Masked loss — every example is  <chat template + instruction> + <response>
  + <eos>.  labels are -100 on everything up to the response, so the model is
  graded ONLY on generating the cordel, never on echoing the instruction.

Runs on GPU (Colab T4: ~10 min) or CPU (an 8 GB laptop: hours — use Colab;
see notebooks/train_sft.ipynb).

    python scripts/make_sft_data.py
    python scripts/sft.py
    python scripts/chat.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE_MODEL = "TucanoBR/Tucano-1b1-Instruct"
BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "..", "model", "cordel-sft-lora")
DATA_PATHS = [
    os.path.join(BASE, "..", "data", "sft.jsonl"),
    os.path.join(BASE, "..", "data", "sft_synth.jsonl"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- config (defaults tuned for a Colab T4 GPU) ----------------------
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

batch_size = int(os.environ.get("SFT_BATCH", 4 if device == "cuda" else 1))
max_len = int(os.environ.get("SFT_MAXLEN", 320))
learning_rate = 2e-4
epochs = float(os.environ.get("SFT_EPOCHS", 2))
eval_every = int(os.environ.get("SFT_EVAL_EVERY", 40))
# ---------------------------------------------------------------------

torch.manual_seed(1337)
if device == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

DEMO_PROMPTS = [
    "Glose o mote: «Quem não arrisca, não petisca.»",
    "Faça uma estrofe de cordel sobre a lua no sertão.",
]


def _ids(x):
    """apply_chat_template can return a list, a tokenizers.Encoding, or a BatchEncoding."""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    if hasattr(x, "ids"):
        x = x.ids
    if x and isinstance(x[0], (list, tuple)):   # nested [[...]]
        x = x[0]
    return [int(t) for t in x]


def build_example(tok, prompt, response):
    """Chat-template the pair; return (input_ids, labels) with the prompt masked."""
    full = _ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False,
    ))
    prefix = _ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
    ))
    if full[: len(prefix)] != prefix:          # template quirk — fall back
        prefix = full[: max(1, len(full) // 3)]
    labels = [-100] * len(prefix) + full[len(prefix):]
    return full[:max_len], labels[:max_len]


def collate(rows, pad_id):
    maxlen = max(len(r[0]) for r in rows)
    x = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    y = torch.full((len(rows), maxlen), -100, dtype=torch.long)
    m = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, (ids, labels) in enumerate(rows):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        m[i, : len(ids)] = 1
    return x.to(device), y.to(device), m.to(device)


def chat_inputs(tok, prompt):
    """A prompt -> {input_ids, attention_mask} tensors on `device`.
    apply_chat_template may hand back a bare tensor or a BatchEncoding."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    ids = ids.to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@torch.no_grad()
def demo(model, tok, n_tokens=70):
    model.eval()
    for p in DEMO_PROMPTS:
        enc = chat_inputs(tok, p)
        out = model.generate(**enc, max_new_tokens=n_tokens, do_sample=True,
                             temperature=0.7, top_k=50, repetition_penalty=1.2,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  › {p}\n    " + text.replace("\n", "\n    ")[:320], flush=True)
    model.train()


def main():
    print(f"device: {device}", flush=True)
    print(f"loading {BASE_MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)

    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=lora_targets, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    pairs = []
    for path in DATA_PATHS:
        if os.path.exists(path):
            pairs += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        else:
            print(f"  (missing {os.path.basename(path)} — run scripts/make_sft_data.py)")
    if not pairs:
        sys.exit("No SFT data.")
    examples = [build_example(tok, p["prompt"], p["response"]) for p in pairs]
    steps = max(1, int(len(examples) * epochs / batch_size))
    print(f"{len(examples)} examples · batch {batch_size} · {epochs} epochs = {steps} steps",
          flush=True)

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=learning_rate)
    model.train()

    order, cursor, t0, running = [], 0, time.time(), 0.0
    for step in range(steps):
        if cursor + batch_size > len(order):
            order = torch.randperm(len(examples)).tolist()
            cursor = 0
        rows = [examples[i] for i in order[cursor: cursor + batch_size]]
        cursor += batch_size

        x, y, m = collate(rows, tok.pad_token_id)
        loss = model(input_ids=x, attention_mask=m, labels=y).loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optim.step()
        running += loss.item()

        if step % eval_every == 0 or step == steps - 1:
            avg = running / (eval_every if step else 1)
            running = 0.0
            print(f"\nstep {step}/{steps}  loss {avg:.4f}  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
            demo(model, tok)
            model.save_pretrained(OUT_DIR)

    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"\nDone. LoRA adapter -> {OUT_DIR}  (try: python scripts/chat.py)", flush=True)


if __name__ == "__main__":
    main()
