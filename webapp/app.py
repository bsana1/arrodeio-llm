"""
arrodeio-llm — a small FastAPI backend serving all three trained stages of the
project side by side, so people can feel the progression themselves.

    v0  from scratch     model_lib/gpt.py + models_v0/checkpoint.pt   (this repo)
    v1  GPT-2 pt + cordel  pierreguillou/gpt2-small-portuguese, fine-tuned
    v2  Tucano + SFT       TucanoBR/Tucano-1b1-Instruct + a LoRA adapter

v1 and v2 load from the Hugging Face Hub by default (set via env vars below) —
override with a local path for development. Models load lazily, on first
request per version, and stay resident after that.

Run locally:
    V1_MODEL=../model/cordel-ft V2_ADAPTER=../model/cordel-sft-lora \
        ../venv/bin/uvicorn app:app --reload --port 7860
"""

import os
import sys
import threading
import time

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # for model_lib

TUCANO_BASE = "TucanoBR/Tucano-1b1-Instruct"
V1_MODEL = os.environ.get("V1_MODEL", "bsana1/arrodeio-gpt2-cordel")
V2_ADAPTER = os.environ.get("V2_ADAPTER", "bsana1/arrodeio-tucano-cordel-lora")

MAX_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 110))

META = {
    "v0": {
        "label": "v0 · do zero",
        "desc": "TinyGPT treinado do zero, só com o corpus de cordel (~3.6M "
                "parâmetros). Não conhece o idioma — espere não-palavras.",
        "kind": "completion",
        "speed": "rápido",
    },
    "v1": {
        "label": "v1 · GPT-2 português + cordel",
        "desc": "GPT-2 português (124M, pierreguillou/gpt2-small-portuguese) "
                "com um fine-tune de estilo cordel.",
        "kind": "completion",
        "speed": "médio (alguns segundos)",
    },
    "v2": {
        "label": "v2 · Tucano + SFT (mais recente)",
        "desc": "Tucano-1b1-Instruct (1.1B, PT-BR nativo) com um adapter LoRA "
                "treinado em pares mote → glosa. Segue instruções.",
        "kind": "instruction",
        "speed": "lento (~1 min em CPU)",
    },
}

app = FastAPI(title="arrodeio-llm")

_models = {}
_load_lock = threading.Lock()
_gen_lock = threading.Lock()   # one generation at a time — this is a free CPU box


def _load_v0():
    from model_lib.gpt import TinyGPT
    from model_lib import tokenizer as tk
    ckpt = torch.load(os.path.join(HERE, "models_v0", "checkpoint.pt"), map_location="cpu")
    tok = tk.from_state(ckpt["tokenizer"])
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {"tok": tok, "model": model}


def _load_v1():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(V1_MODEL)
    model = AutoModelForCausalLM.from_pretrained(V1_MODEL)
    model.eval()
    return {"tok": tok, "model": model}


def _load_v2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    # bf16, not fp32: on a memory-constrained CPU box fp32 (2x the RAM) pushes
    # into swap and generation goes from ~1.5 tok/s to ~0.1 tok/s. Measured.
    tok = AutoTokenizer.from_pretrained(TUCANO_BASE)
    model = AutoModelForCausalLM.from_pretrained(TUCANO_BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, V2_ADAPTER)
    model.eval()
    return {"tok": tok, "model": model}


LOADERS = {"v0": _load_v0, "v1": _load_v1, "v2": _load_v2}


def get_model(version):
    with _load_lock:
        if version not in _models:
            print(f"[arrodeio] loading {version} ...", flush=True)
            t0 = time.time()
            _models[version] = LOADERS[version]()
            print(f"[arrodeio] {version} loaded in {time.time()-t0:.0f}s", flush=True)
    return _models[version]


FEWSHOT_V1 = (
    "Eram doze cavaleiros\nhomens muito valorosos,\ndestemidos, animosos,\n"
    "entre todos os guerreiros.\n\n"
)


def run_v0(prompt):
    m = get_model("v0")
    seed = prompt.strip() or "Vou contar uma história"
    ids = torch.tensor([m["tok"].encode(seed)])
    out = m["model"].generate(ids, MAX_TOKENS, temperature=0.8, top_k=30)
    return m["tok"].decode(out[0].tolist())


def run_v1(prompt):
    m = get_model("v1")
    theme = prompt.strip() or "a vida no sertão"
    seed = FEWSHOT_V1 + theme + "\n"
    ids = m["tok"](seed, return_tensors="pt").input_ids
    out = m["model"].generate(ids, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.8,
                              top_k=40, repetition_penalty=1.3, pad_token_id=m["tok"].eos_token_id)
    return m["tok"].decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def run_v2(prompt):
    m = get_model("v2")
    text = prompt.strip() or "Faça uma estrofe de cordel sobre a saudade."
    enc = m["tok"].apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    out = m["model"].generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                              max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7,
                              top_k=50, repetition_penalty=1.2, pad_token_id=m["tok"].eos_token_id)
    return m["tok"].decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


RUNNERS = {"v0": run_v0, "v1": run_v1, "v2": run_v2}


class GenRequest(BaseModel):
    version: str
    prompt: str = ""


@app.get("/api/versions")
def versions():
    return META


@app.post("/api/generate")
def generate(req: GenRequest):
    if req.version not in RUNNERS:
        return {"error": f"unknown version {req.version!r}"}
    t0 = time.time()
    with _gen_lock:
        try:
            text = RUNNERS[req.version](req.prompt)
        except Exception as e:
            return {"error": str(e)}
    return {"text": text, "elapsed_s": round(time.time() - t0, 1)}


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
