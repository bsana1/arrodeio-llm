"""
Phase 2c: better SFT data — topics that actually describe the verses.

v1 problem (scripts/make_sft_data.py): the instruction for each real stanza was
guessed from keyword regexes, with a RANDOM fallback theme when nothing matched.
So many pairs were "write about the sea" -> a stanza about a patroa. The model
learned the cordel shape but not to stay on topic.

v2 fixes the prompt side and the unit of training:
  - Each response is 1 or 2 consecutive, adjacent stanzas of a real folheto
    (verbatim), so ideas have to carry across stanzas.
  - The topic of every chunk was written by hand (by Claude, reading each one;
    data/sft_v2_topics.jsonl) — teacher labeling. We first tried having Tucano
    1.1B write them (--llm), but it mostly paraphrases the first verse
    ("após ter apanhado,"), which makes ungrammatical prompts.

    python scripts/make_sft_data_v2.py        # writes data/sft_v2.jsonl (no GPU needed)
    python scripts/make_sft_data_v2.py --llm  # re-label with Tucano instead (GPU recommended)
    SFT_DATA=data/sft.jsonl,data/sft_v2.jsonl python scripts/sft.py
"""

import glob
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(1337)

BASE = os.path.dirname(__file__)
CORDEL_DIR = os.path.join(BASE, "..", "data", "cordel")
OUT = os.path.join(BASE, "..", "data", "sft_v2.jsonl")
LABELER = "TucanoBR/Tucano-1b1-Instruct"

MAX_PER_FOLHETO = 60          # keep the big folhetos from dominating
BATCH = 8 if torch.cuda.is_available() else 1
device = "cuda" if torch.cuda.is_available() else "cpu"

ASK = ("Leia estes versos de cordel e diga, em no máximo 8 palavras, qual é o "
       "assunto deles, como uma expressão nominal (começando por um substantivo "
       "ou artigo, nunca por um verbo nem copiando um trecho). Responda só com o assunto.\n\n{verses}")

# Few-shot (verses, topic) so the 1.1B labeler imitates the format instead of
# copying a phrase out of the verses.
FEWSHOT = [
    ("Seca as terras as folhas caem,\nMorre o gado sai o povo,\nO vento varre a campina,\n"
     "Rebenta a seca de novo;\nCinco, seis mil emigrantes\nFlagelados retirantes\n"
     "Vagam mendigando o pão,\nAcabam-se os animais\nFicando limpo os currais\nOnde houve a criação.",
     "a seca e a fuga dos retirantes"),
    ("Eu juro pelo meu rifle,\nQue o Padre José Paulino\nCai sempre na ratoeira\n"
     "E paga o grosso e o fino,\nNão há de casar mais homem,\nNem batizar mais menino.",
     "a ameaça de um cangaceiro a um padre"),
    ("Vê-se uma mãe cadavérica\nQue já não pode falar,\nEstreitando o filho ao peito\n"
     "Sem o poder consolar\nLança-lhe um olhar materno\nSoluça implora ao Eterno\n"
     "Invoca da Virgem o nome\nEla débil triste e louca\nApenas beija-lhe a boca\nE ambos morrem de fome.",
     "a morte de uma mãe e seu filho de fome"),
]

INSTR_1 = [
    "Escreva uma estrofe de cordel sobre {t}.",
    "Faça uma estrofe de cordel sobre {t}.",
    "Componha uma sextilha de cordel sobre {t}.",
    "No estilo do folheto, escreva uns versos sobre {t}.",
]
INSTR_2 = [
    "Escreva um cordel de duas estrofes sobre {t}.",
    "Faça duas estrofes de cordel sobre {t}.",
    "Componha dois trechos de folheto, em sequência, sobre {t}.",
]


def blocks(path):
    """Yield stanza text, or None for a block that isn't a usable stanza, in file order."""
    text = open(path, encoding="utf-8").read()
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        joined = "\n".join(lines)
        ok = 3 <= len(lines) <= 12 and 40 <= len(joined) <= 700 \
            and max((len(l) for l in lines), default=0) < 90
        out.append(joined if ok else None)
    return out


def windows(path):
    """All 1-stanza and non-overlapping adjacent 2-stanza chunks of a folheto."""
    bl = blocks(path)
    res = [b for b in bl if b]
    i = 0
    while i + 1 < len(bl):
        if bl[i] and bl[i + 1]:
            res.append(bl[i] + "\n\n" + bl[i + 1])
            i += 2
        else:
            i += 1
    if len(res) > MAX_PER_FOLHETO:
        res = random.sample(res, MAX_PER_FOLHETO)
    return res


def clean_topic(raw):
    t = raw.strip().split("\n")[0].strip().strip("\"'«».:;- ")
    t = re.sub(r"^(o assunto( é| dos versos é)?|assunto|tema)\s*[:\-]?\s*", "", t, flags=re.I)
    t = t.strip("\"'«».:;- ")
    words = t.split()
    if not (2 <= len(words) <= 12):
        return None
    if re.search(r"versos|estrofe|cordel|poema", t, re.I):    # labeler described the form, not the content
        return None
    # fragments copied from the verses ("que se dizem cristãos", "era João Batista",
    # "oliveiros disse") rather than a noun phrase naming the subject
    if words[0].lower() in {"que", "era", "é", "foi", "são", "e", "mas", "se", "com", "pois", "como", "quando", "onde", "ao", "já", "não"}:
        return None
    if words[-1].lower().strip(",.") in {"disse", "diz", "respondeu", "falou", "dizendo", "que", "e", "de", "da", "do"}:
        return None
    return t[0].lower() + t[1:]


@torch.no_grad()
def label(tok, model, verse_list):
    topics = []
    for i in range(0, len(verse_list), BATCH):
        batch = verse_list[i:i + BATCH]
        shots = []
        for sv, st in FEWSHOT:
            shots += [{"role": "user", "content": ASK.format(verses=sv)},
                      {"role": "assistant", "content": st}]
        prompts = [tok.apply_chat_template(
            shots + [{"role": "user", "content": ASK.format(verses=v)}],
            tokenize=False, add_generation_prompt=True) for v in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for row in out:
            topics.append(clean_topic(tok.decode(row[enc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)))
        print(f"  labeled {min(i + BATCH, len(verse_list))}/{len(verse_list)}", flush=True)
    return topics


TOPICS = os.path.join(BASE, "..", "data", "sft_v2_topics.jsonl")


def load_written_topics():
    """(verses, topic) pairs whose topics were written by hand (by Claude, reading each chunk)."""
    rows = [json.loads(l) for l in open(TOPICS, encoding="utf-8")]
    return [r["verses"] for r in rows], [r["topic"] for r in rows]


def main():
    import sys
    if "--llm" in sys.argv:        # re-label with Tucano itself (weak: mostly paraphrases the first line)
        chunks = []
        for path in sorted(glob.glob(os.path.join(CORDEL_DIR, "*.txt"))):
            chunks += windows(path)
        random.shuffle(chunks)
        print(f"{len(chunks)} chunks; labeling with {LABELER} on {device}")
        tok = AutoTokenizer.from_pretrained(LABELER)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            LABELER, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        model.eval()
        topics = label(tok, model, chunks)
    else:
        chunks, topics = load_written_topics()
        print(f"{len(chunks)} chunks with written topics ({os.path.relpath(TOPICS)})")

    kept, dropped = [], 0
    for verses, topic in zip(chunks, topics):
        if not topic:
            dropped += 1
            continue
        two = "\n\n" in verses
        prompt = random.choice(INSTR_2 if two else INSTR_1).format(t=topic)
        kept.append({"prompt": prompt, "response": verses})

    with open(OUT, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} pairs to {os.path.relpath(OUT)} (dropped {dropped} unusable topics)")
    for p in kept[:6]:
        print("  •", p["prompt"])


if __name__ == "__main__":
    main()
