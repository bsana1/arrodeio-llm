"""
arrodeio-llm — Gradio + ZeroGPU version.

Same idea as ../webapp (v0/v1/v2 side by side), rebuilt on the Gradio SDK
because that's what free Hugging Face accounts can host on real (shared) GPU
hardware for $0 — Docker/FastAPI Spaces on CPU-basic now require PRO.

v0 and v1 are small enough that CPU is fine and they don't touch the GPU
quota. v2 (Tucano, 1.1B) is wrapped in @spaces.GPU, which borrows a shared
GPU for the duration of the call — per ZeroGPU's rules, the model has to be
moved to 'cuda' at *module* level (not lazily inside the decorated function),
which only works inside an actual ZeroGPU Space. Wrapped in try/except so
this file still imports and v0/v1 still work when run locally on a machine
with no CUDA build of torch (developing on a Mac, for instance).
"""

import os
import sys

# `spaces` must be imported before `torch` (ZeroGPU docs): it patches torch's
# CUDA internals, and that patch has to be in place before anything else
# (including the module-level .to("cuda") below) touches CUDA — otherwise the
# real GPU handoff when @spaces.GPU actually forks a worker fails with
# "RuntimeError: No CUDA GPUs are available" even though the Space's hardware
# is correctly zero-a10g.
import spaces
import torch
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model_lib.gpt import TinyGPT
from model_lib import tokenizer as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TUCANO_BASE = "TucanoBR/Tucano-1b1-Instruct"
V1_MODEL = os.environ.get("V1_MODEL", "bsana1/arrodeio-gpt2-cordel")
V2_ADAPTER = os.environ.get("V2_ADAPTER", "bsana1/arrodeio-tucano-cordel-lora")
MAX_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 110))

# ---- v0: from-scratch TinyGPT (CPU, instant) --------------------------
_ckpt = torch.load(os.path.join(HERE, "models_v0", "checkpoint.pt"), map_location="cpu")
_v0_tok = tk.from_state(_ckpt["tokenizer"])
_v0_model = TinyGPT(**_ckpt["config"])
_v0_model.load_state_dict(_ckpt["model_state"])
_v0_model.eval()

# ---- v1: GPT-2 pt + cordel style-tune (CPU, a few seconds) ------------
_v1_tok = AutoTokenizer.from_pretrained(V1_MODEL)
_v1_model = AutoModelForCausalLM.from_pretrained(V1_MODEL)
_v1_model.eval()

# ---- v2: Tucano + LoRA (GPU via ZeroGPU) -------------------------------
# Load everything on CPU first, explicitly. ZeroGPU's emulation layer makes
# torch.cuda.is_available() report True even at module level (outside any
# @spaces.GPU call), so PEFT's device auto-inference (peft.utils.infer_device,
# used inside load_adapter -> load_peft_weights) reaches for a real GPU that
# doesn't exist yet here and crashes with "No CUDA GPUs are available".
# `device_map` (a from_pretrained/base-model concept) does NOT control this —
# load_adapter's own parameter is `torch_device`. Pass that explicitly; we
# move the whole assembled model to 'cuda' ourselves, afterward.
_v2_tok = AutoTokenizer.from_pretrained(TUCANO_BASE)
_v2_model = AutoModelForCausalLM.from_pretrained(TUCANO_BASE, dtype=torch.bfloat16)
_v2_model = PeftModel.from_pretrained(_v2_model, V2_ADAPTER, torch_device="cpu")
_v2_model.eval()
try:
    _v2_model = _v2_model.to("cuda")   # required at module level per ZeroGPU docs
    V2_DEVICE = "cuda"
except Exception:
    V2_DEVICE = "cpu"                  # local dev machine with no CUDA torch build

FEWSHOT_V1 = (
    "Eram doze cavaleiros\nhomens muito valorosos,\ndestemidos, animosos,\n"
    "entre todos os guerreiros.\n\n"
)


def run_v0(prompt):
    seed = (prompt or "").strip() or "Vou contar uma história"
    ids = torch.tensor([_v0_tok.encode(seed)])
    out = _v0_model.generate(ids, MAX_TOKENS, temperature=0.8, top_k=30)
    return _v0_tok.decode(out[0].tolist())


def run_v1(prompt):
    theme = (prompt or "").strip() or "a vida no sertão"
    seed = FEWSHOT_V1 + theme + "\n"
    ids = _v1_tok(seed, return_tensors="pt").input_ids
    out = _v1_model.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.8,
                             top_k=40, repetition_penalty=1.3, pad_token_id=_v1_tok.eos_token_id)
    return _v1_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@spaces.GPU(duration=60)
def run_v2(prompt):
    text = (prompt or "").strip() or "Faça uma estrofe de cordel sobre a saudade."
    enc = _v2_tok.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    ids = ids.to(V2_DEVICE)
    out = _v2_model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7,
                             top_k=50, repetition_penalty=1.2, pad_token_id=_v2_tok.eos_token_id)
    return _v2_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


VERSIONS = {
    "v0 · do zero": run_v0,
    "v1 · GPT-2 português + cordel": run_v1,
    "v2 · Tucano + SFT (mais recente)": run_v2,
}
DESCS = {
    "v0 · do zero": "TinyGPT treinado do zero, só com o corpus de cordel (~3.6M "
                    "parâmetros). Ainda não aprendeu o idioma — o texto sai cheio "
                    "de palavras inventadas. · instantâneo",
    "v1 · GPT-2 português + cordel": "GPT-2 português (124M) com um fine-tune de "
                                     "estilo cordel. · alguns segundos",
    "v2 · Tucano + SFT (mais recente)": "Tucano-1b1-Instruct (1.1B, PT-BR nativo) + "
                                        "LoRA treinado em pares mote → glosa. Roda em "
                                        "GPU real (ZeroGPU). · alguns segundos",
}
DEFAULT_VERSION = "v2 · Tucano + SFT (mais recente)"

CSS = """
.gradio-container { font-family: "SF Mono","Menlo","Consolas",monospace !important; }
#output textarea { font-family: inherit !important; }
"""


def generate(version, prompt):
    if not prompt or not prompt.strip():
        return "(escreva um mote ou tema primeiro)"
    return VERSIONS[version](prompt)


with gr.Blocks(title="arrodeio-llm", css=CSS, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        "# 🪕 arrodeio-llm\n"
        "Um LLM em português brasileiro que escreve *cordel* — a poesia rimada "
        "e popular do Nordeste do Brasil.\n\n"
        "Projeto de aprendizado de **[Bernardo Sana](https://huggingface.co/bsana1)** "
        "para explorar como os dados de treinamento afetam a qualidade do modelo "
        "nas etapas de pré-treinamento, fine-tuning e inferência. Está na versão 2, "
        "construída sobre o [Tucano](https://huggingface.co/TucanoBR/Tucano-1b1-Instruct) "
        "com um adaptador LoRA "
        "([modelo v2](https://huggingface.co/bsana1/arrodeio-tucano-cordel-lora)); "
        "a v1 usa um [GPT-2 em português](https://huggingface.co/bsana1/arrodeio-gpt2-cordel) "
        "com fine-tune de estilo.\n\n"
        "Escolha uma versão (0, 1 ou 2), dê um tema ou mote (ex: *\"água mole em "
        "pedra dura\"*) e veja o cordel que ele escreve.\n\n"
        "[código no GitHub](https://github.com/bsana1/arrodeio-llm)"
    )
    version = gr.Radio(list(VERSIONS.keys()), value=DEFAULT_VERSION, label="versão")
    desc = gr.Markdown(DESCS[DEFAULT_VERSION])
    version.change(lambda v: DESCS[v], inputs=version, outputs=desc)

    prompt = gr.Textbox(
        label="mote ou tema",
        placeholder="ex: Glose o mote: «Água mole em pedra dura.»",
        lines=2,
    )
    btn = gr.Button("gerar", variant="primary")
    output = gr.Textbox(label="resposta", lines=8, elem_id="output")

    btn.click(generate, inputs=[version, prompt], outputs=output)
    prompt.submit(generate, inputs=[version, prompt], outputs=output)

if __name__ == "__main__":
    demo.launch()
