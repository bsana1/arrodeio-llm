"""
Compare LoRA adapters on the same prompts (GPU recommended; made for Colab).

Each candidate is (label, base model, adapter path-or-hub-id). Loads one at a
time so a 3.7B and a 1.1B model never share GPU memory.

    python scripts/compare_adapters.py \
        --new model/cordel-sft-lora-t2-3b7 --new-base Polygl0t/Tucano2-qwen-3.7B-Instruct \
        --old bsana1/arrodeio-tucano-cordel-lora --old-base TucanoBR/Tucano-1b1-Instruct
"""

import argparse
import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Escreva um cordel sobre o mar.",
    "Faça uma estrofe de cordel sobre a saudade.",
    "Escreva um cordel sobre a seca no sertão.",
    "Escreva um cordel de duas estrofes sobre um vaqueiro e seu boi.",
    "Escreva um cordel sobre: batatinha quando nasce.",
    "Escreva um cordel sobre: quem não arrisca, não petisca.",
    "Escreva um cordel sobre um pobre que enganou um duque rico.",
    "Escreva um cordel sobre a feira de domingo.",
]


def run(label, base, adapter, seeds, settings):
    print(f"\n{'#' * 70}\n# {label}: {base} + {adapter}\n{'#' * 70}", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, adapter).eval()
    for temp, rep in settings:
        print(f"\n{'~' * 70}\n~ sampling: temperature={temp} repetition_penalty={rep}\n{'~' * 70}",
              flush=True)
        for p in PROMPTS:
            print(f"\n===== {p}", flush=True)
            enc = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                          add_generation_prompt=True, return_tensors="pt")
            ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
            for seed in seeds:
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=200, do_sample=True, temperature=temp,
                                         top_k=50, repetition_penalty=rep,
                                         pad_token_id=tok.eos_token_id)
                print(f"--- {label} #{seed}\n" + tok.decode(out[0][ids.shape[1]:],
                                                           skip_special_tokens=True).strip(),
                      flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-base", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-base", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--settings", default="0.7:1.2",
                    help="comma list of temperature:repetition_penalty, e.g. 0.7:1.2,0.5:1.05")
    ap.add_argument("--only-new", action="store_true", help="skip the --old model")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    settings = [(float(t), float(r)) for t, r in (x.split(":") for x in a.settings.split(","))]
    if not a.only_new:
        run("v2.1", a.old_base, a.old, seeds, settings)
    run("NEW", a.new_base, a.new, seeds, settings)
