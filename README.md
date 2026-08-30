# arrodeio-llm 🇧🇷

> *arrodeio* (nordestino) — jeito de falar dando voltas, sem chegar logo ao ponto.
> É mais ou menos o que esse modelo faz.

Um projeto de aprendizado sobre LLMs, feito com **ditados populares brasileiros**
e **cordel** de domínio público. Dois modelos:

1. um **GPT treinado do zero** (char/BPE), para entender na prática como o
   pré-treino funciona — e por que corpus pequeno não basta;
2. um **GPT-2 português pré-treinado, ajustado (fine-tuned)** no mesmo corpus,
   que de fato gera cordel legível.

A hands-on LLM learning project built on public-domain Brazilian folk sayings
(*ditados*) and *cordel* poetry. It contains **two models**:

1. a **from-scratch GPT** (char- or BPE-level, ~1–4M params) — to make the
   mechanics of pretraining concrete, and to feel firsthand why a tiny corpus
   can't teach a language;
2. a **fine-tuned pretrained Portuguese GPT-2** (124M params) — the one that
   actually produces readable cordel.

Everything runs on a laptop CPU.

## The two tracks

|  | **from scratch** (Phase 1) | **fine-tuned** (Phase 1b) |
|---|---|---|
| starting point | random weights | `pierreguillou/gpt2-small-portuguese` |
| what it teaches | tokenization, the training loop, autograd, attention, sampling, **scaling laws** | transfer learning, catastrophic forgetting, why *loss ≠ goal* |
| trains in | ~10–30 min | ~25 min |
| build | `train_tokenizer.py` → `train.py` | `finetune.py` |
| sample with | `generate.py` | `generate_ft.py` |
| output | cordel-shaped **non-words** — `"Quem áis trame vencem"` | real Portuguese cordel — `"Era uma vez um cangaceiro; / uma alma desmarcada"` |

The from-scratch model is **not** a foundation the fine-tuned one builds on — they
share only `data/corpus.txt`. Phase 1 is the "how it works" exhibit; Phase 1b is
the working generator.

### Why keep the from-scratch model at all?

Because running it is the lesson. With ~150 KB of text (≈ 45k tokens) a
from-scratch model **cannot** learn which letter-sequences are real Portuguese
words — no model size, vocab, or sampling trick fixes a 250,000× data gap versus
a real pretraining run. Watching its `sample:` line go gibberish → cordel-flavored
nonsense → verbatim memorization, while train loss drops and val loss climbs, is
the clearest way to *see* why pretraining needs the scale it does.

## Project structure

```
arrodeio-llm/
├── data/
│   ├── ditados.txt        # ~90 Brazilian folk sayings, one per line
│   ├── cordel.txt          # sourcing notes (verses live in cordel/)
│   ├── cordel/             # 17 public-domain folhetos, fetched from Wikisource
│   └── corpus.txt           # generated — ditados + cordel merged
├── model/
│   ├── gpt.py               # TinyGPT architecture (nanoGPT-style)
│   ├── tokenizer.py         # char + byte-level BPE tokenizers
│   ├── tokenizer.json       # generated — the trained BPE vocab
│   ├── checkpoint.pt        # generated — the from-scratch weights
│   └── cordel-ft/           # generated — the fine-tuned model (~500 MB)
├── scripts/
│   ├── fetch_cordel.py      # pull public-domain cordel from pt.wikisource.org
│   ├── prepare_data.py      # data/**/*.txt -> data/corpus.txt
│   ├── train_tokenizer.py   # data/corpus.txt -> model/tokenizer.json (BPE)
│   ├── train.py             # Phase 1  — pretrain TinyGPT from scratch
│   ├── generate.py          # inference for the from-scratch model
│   ├── finetune.py          # Phase 1b — style fine-tune the pretrained GPT-2
│   └── generate_ft.py       # inference for the fine-tuned model
└── README.md
```

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_cordel.py       # (optional) refresh data/cordel/ from Wikisource
python scripts/prepare_data.py       # build data/corpus.txt
```

### Track 1 — from scratch

```bash
python scripts/train_tokenizer.py    # data/corpus.txt -> model/tokenizer.json
python scripts/train.py               # ~10–30 min CPU
python scripts/generate.py "Quem"     # expect cordel-shaped non-words — that's the point
```

`train_tokenizer.py` builds a byte-level BPE vocab (GPT-2 style): frequent chunks
like `" que"` / `"ção"` become single tokens, so sequences are ~2.5× shorter and
samples are stitched from real word-pieces. `--kind char` gives the original
one-token-per-character setup; `--merges N` sets vocab size (`256 + N`).
`train.py` keeps the **lowest-val-loss** checkpoint (the corpus is so small the
model memorizes it well before `max_iters`).

### Track 2 — fine-tune a pretrained model

```bash
# transformers is already in requirements.txt; first run downloads ~500 MB
python scripts/finetune.py                          # ~25 min CPU
python scripts/generate_ft.py "Vou contar uma história"
python scripts/generate_ft.py "A donzela" --base    # hear the un-tuned model for contrast
python scripts/generate_ft.py "A donzela" --raw     # skip the few-shot priming stanzas
```

`finetune.py` loads `pierreguillou/gpt2-small-portuguese`, freezes the lower
layers, and continues training the top 6 transformer blocks on the corpus
(plain next-token prediction — *continued pretraining*, not instruction tuning).

It deliberately does **not** keep the lowest-val-loss checkpoint: for style
transfer, val loss bottoms out in ~75 steps while the cordel voice keeps
sharpening for hundreds more. It runs to a tuned `max_iters` and keeps a rolling
save. `generate_ft.py` primes the model with two real stanzas by default
(`--raw` to disable) to keep it from sliding back into encyclopedia prose.

### Track 2, Phase 2 — SFT with LoRA (needs a GPU)

`gpt2-small-portuguese` is a *completion* model — it never learned to follow an
instruction, and no dataset we could hand-build taught it to (see git history for
three failed attempts). Phase 2 switches to **`TucanoBR/Tucano-1b1-Instruct`**
(1.1B, native Brazilian Portuguese, already SFT+DPO'd by PUCRS) and adds cordel
style + the *mote e glosa* task on top.

- **Data:** `data/sft.jsonl` (50 hand-written `mote → glosa` pairs) +
  `data/sft_synth.jsonl` (build with `scripts/make_sft_data.py` — *instruction
  backtranslation*: real stanzas kept as verbatim responses, plausible
  instructions fabricated).
- **`scripts/sft.py`:** LoRA (freeze the 1.1B base, train ~4.5M adapter params) +
  masked loss (labels `-100` on the instruction tokens) + the chat template.
- **`scripts/chat.py`:** a REPL. `--base` talks to raw Tucano for contrast.

A 1.1B model won't fine-tune on a small laptop (an 8 GB CPU machine needs ~90 s
*per example*). Run it on a free GPU instead:

**[▶ Open `notebooks/train_sft.ipynb` in Colab](https://colab.research.google.com/github/bsana1/arrodeio-llm/blob/main/notebooks/train_sft.ipynb)** — ~10 min on a free T4, then download the ~9 MB adapter into `model/cordel-sft-lora/` and run `chat.py` locally (slow inference, but it works).

## Contributing to the dataset

More material makes both models better (and the fine-tuned one noticeably so).
Welcome:

- **Ditados populares** — traditional / anonymous folk sayings only, no modern
  copyrighted quotes. One per line in `data/ditados.txt`.
- **Public-domain cordel** — add Wikisource page titles to `DEFAULT_WORKS` in
  `scripts/fetch_cordel.py`. Authors must be long-dead (Leandro Gomes de Barros
  d. 1918, Silvino Pirauá de Lima d. 1913, etc.). See `data/cordel.txt`.
- **Regional ditado variants** — note the region if you know it.

## Roadmap

- [x] Phase 1 — pretrain from scratch on ditados + cordel (mechanics; output
      stays noisy on a corpus this small)
- [x] Phase 1b — style fine-tune a pretrained PT GPT-2 for readable output
- [~] Phase 2 — SFT with LoRA on Tucano-1b1-Instruct: masked-loss instruction
      tuning for the *mote e glosa* task (`scripts/sft.py`, run on a free GPU
      via `notebooks/train_sft.ipynb`)
- [ ] Phase 3 — preference pairs + a tiny reward model
- [ ] Phase 4 — best-of-N reranking / rejection-sampling fine-tune
- [x] Grow the corpus with real public-domain cordel (`scripts/fetch_cordel.py`
      — 17 folhetos, ~150k chars, from pt.wikisource.org)
- [ ] Community-contributed regional ditados

Phases 2–4 build on the **fine-tuned** model (`model/cordel-ft/`) — the same
training loop, just different data and a masked loss. They'd work on the
from-scratch model too, but you'd be watching loss curves, not readable output.

## License

Code: MIT. Dataset: traditional / public-domain folk material only — see
`data/cordel.txt` before adding any source. The fine-tuned model inherits the
license of `pierreguillou/gpt2-small-portuguese` (MIT).
