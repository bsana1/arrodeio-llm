"""
Run a fixed prompt set through every stage of the project and save the results
for the side-by-side comparison (repo doc + the web UI).

Stages:
  1  scratch    — from-scratch TinyGPT (model/checkpoint.pt)          [completion]
  2  gpt2       — pierreguillou/gpt2-small-portuguese, un-tuned       [completion]
  3  gpt2-ft    — that model + our cordel style fine-tune (Phase 1b)  [completion]
  4  tucano     — TucanoBR/Tucano-1b1-Instruct, un-tuned              [instruction]
  5  tucano-sft — Tucano + our cordel LoRA (Phase 2)                  [instruction]

Stages 1-3 run in seconds on a laptop. Stages 4-5 need a GPU — run this on Colab
with `--stages 4,5` (see notebooks/). Each run merges into samples/comparison.json.

    python scripts/collect_samples.py --stages 1,2,3
    python scripts/collect_samples.py --stages 4,5        # on a GPU
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
import torch

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "samples", "comparison.json")
OUT_MD = os.path.join(BASE, "..", "samples", "comparison.md")

ITEMS = [
    {"theme": "a seca no sertão", "mote": "Água mole em pedra dura, tanto bate até que fura."},
    {"theme": "um cangaceiro valente", "mote": "Quem não arrisca, não petisca."},
    {"theme": "a viola do cantador", "mote": "Quem canta seus males espanta."},
]

STAGE_META = {
    "scratch":    {"n": 1, "kind": "completion",  "label": "do zero (char/BPE GPT ~2M)"},
    "gpt2":       {"n": 2, "kind": "completion",  "label": "GPT-2 português, cru (124M)"},
    "gpt2-ft":    {"n": 3, "kind": "completion",  "label": "GPT-2 português + estilo cordel"},
    "tucano":     {"n": 4, "kind": "instruction", "label": "Tucano-1b1-Instruct, cru"},
    "tucano-sft": {"n": 5, "kind": "instruction", "label": "Tucano + SFT cordel (arrodeio-llm)"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_json():
    if os.path.exists(OUT):
        return json.load(open(OUT, encoding="utf-8"))
    return {"items": ITEMS, "stages": {}}


def save_json(data):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    render_md(data)


def render_md(data):
    order = sorted(data["stages"].items(), key=lambda kv: kv[1]["meta"]["n"])
    lines = ["# arrodeio-llm — a mesma pergunta, cada estágio do projeto\n",
             "Os estágios 1–3 *continuam* um texto; os 4–5 *seguem uma instrução*.\n"]
    for it in data["items"]:
        th = it["theme"]
        lines.append(f"\n## {th}\n")
        lines.append(f"*mote:* «{it['mote']}»\n")
        for name, st in order:
            m = st["meta"]
            txt = st["by_theme"].get(th, "").strip()
            lines.append(f"\n**{m['n']}. {m['label']}** — _{m['kind']}_\n")
            lines.append("```\n" + txt + "\n```\n")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---- stage 1: from-scratch TinyGPT ----------------------------------
def run_scratch():
    import sys
    sys.path.insert(0, os.path.join(BASE, ".."))
    from model.gpt import TinyGPT
    from model import tokenizer as tk
    ckpt = torch.load(os.path.join(BASE, "..", "model", "checkpoint.pt"), map_location="cpu")
    tok = tk.from_state(ckpt["tokenizer"])
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    out = {}
    for it in ITEMS:
        seed = "Vou contar uma história"
        idx = torch.tensor([tok.encode(seed)])
        gen = model.generate(idx, 100, temperature=0.8, top_k=30)
        out[it["theme"]] = tok.decode(gen[0].tolist())
    return out


# ---- stages 2 & 3: GPT-2 completion models -------------------------
def run_gpt2(with_ft):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = os.path.join(BASE, "..", "model", "cordel-ft") if with_ft else "pierreguillou/gpt2-small-portuguese"
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src).to(device)
    model.eval()
    out = {}
    for it in ITEMS:
        seed = f"Vou contar uma história sobre {it['theme']}:\n"
        ids = tok(seed, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=110, do_sample=True, temperature=0.8,
                             top_k=40, repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        out[it["theme"]] = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return out


# ---- stages 4 & 5: Tucano instruction models ----------------------
def run_tucano(with_lora):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    BASE_MODEL = "TucanoBR/Tucano-1b1-Instruct"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    if with_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(BASE, "..", "model", "cordel-sft-lora"))
    model.eval()
    out = {}
    for it in ITEMS:
        prompt = f"Glose o mote: «{it['mote']}»"
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50,
                             repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        out[it["theme"]] = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return out


RUNNERS = {
    "scratch": lambda: run_scratch(),
    "gpt2": lambda: run_gpt2(False),
    "gpt2-ft": lambda: run_gpt2(True),
    "tucano": lambda: run_tucano(False),
    "tucano-sft": lambda: run_tucano(True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3",
                    help="comma list of stage numbers or names, e.g. '1,2,3' or 'tucano-sft'")
    args = ap.parse_args()

    by_num = {str(m["n"]): name for name, m in STAGE_META.items()}
    wanted = [by_num.get(s.strip(), s.strip()) for s in args.stages.split(",")]

    data = load_json()
    data["items"] = ITEMS
    for name in wanted:
        if name not in RUNNERS:
            print(f"  ?? unknown stage {name!r}")
            continue
        print(f"running stage {STAGE_META[name]['n']} · {name} ({device}) ...", flush=True)
        results = RUNNERS[name]()
        data["stages"][name] = {"meta": STAGE_META[name], "by_theme": results}
        save_json(data)
        for th, txt in results.items():
            print(f"  [{th}]\n    " + txt.replace("\n", "\n    ")[:280] + "\n")

    print(f"saved -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
