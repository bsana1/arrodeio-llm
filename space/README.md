---
title: arrodeio-llm
emoji: 🪕
colorFrom: yellow
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
license: apache-2.0
short_description: Um LLM brasileiro que escreve cordel, em 3 estágios de aprendizado
---

# arrodeio-llm

A Brazilian-Portuguese LLM that writes *cordel* — the rhyming folk poetry of
the Brazilian Northeast. A learning project by [Bernardo Sana](https://huggingface.co/bsana1)
exploring how training data shapes model quality across pretraining,
fine-tuning, and inference.

Try three stages side by side: a from-scratch char-level GPT (v0), a
fine-tuned Portuguese [GPT-2](https://huggingface.co/bsana1/arrodeio-gpt2-cordel) (v1),
and [Tucano-1b1-Instruct](https://huggingface.co/TucanoBR/Tucano-1b1-Instruct)
fine-tuned with a [LoRA adapter](https://huggingface.co/bsana1/arrodeio-tucano-cordel-lora) (v2).

v2 (the best one) runs on a shared ZeroGPU — genuinely free, no account
upgrade needed, just a daily time quota per visitor.

Code and full write-up: https://github.com/bsana1/arrodeio-llm
