"""
Phase 2b: synthetic SFT data by instruction backtranslation.

We have ~600 real public-domain cordel stanzas in data/cordel/ but no
(instruction, response) pairs. So we keep the RESPONSES authentic — real
stanzas, verbatim — and FABRICATE plausible instructions that would have
produced them:

  - "Continue o cordel: «estrofe N»"     -> estrofe N+1   (always correct)
  - "Faça uma estrofe sobre <tema>"      -> uma estrofe    (tema inferred from
                                                           keywords in the stanza)

Only the prompts are synthetic, and they only need to be roughly right. This is
the technique from Meta's "instruction backtranslation" paper, scaled way down.

    python scripts/make_sft_data.py        # writes data/sft_synth.jsonl
    python scripts/sft.py                    # trains on sft.jsonl + sft_synth.jsonl
"""

import glob
import json
import os
import random
import re

random.seed(1337)

BASE = os.path.dirname(__file__)
CORDEL_DIR = os.path.join(BASE, "..", "data", "cordel")
OUT = os.path.join(BASE, "..", "data", "sft_synth.jsonl")

MAX_PAIRS = 600
MAX_CONTINUATION = 220
MAX_PER_FOLHETO = 55       # keep the big folhetos (Oliveiros!) from dominating

# keyword pattern -> theme label. First match wins; order = priority.
THEMES = [
    (r"cangac|silvino|lampi[ãa]o|volante|jagun[çc]|tenente|peixeira|rifle|cabra da peste", "o cangaceiro e a volante"),
    (r"\b(rei|rainha|pr[ií]ncip|espada|guerreiro|batalha|cavalleiro|cavaleiro|conde|castelo|turco|mouro|oliveiros|ferrabr[áa]s)", "uma batalha dos tempos antigos"),
    (r"diabo|inferno|\balma\b|pecado|tenta[çc][ãa]o|capeta|satan[áa]s|cão coxo", "o diabo e a alma do pecador"),
    (r"padre|santo|reza|missa|igreja|romaria|juazeiro|milagre|bispo|frei\b|ora[çc][ãa]o", "a fé e a religião do povo"),
    (r"\bseca\b|estiagem|inverno que|\bchuva\b|nuvem|trov[ãa]o|sol que racha|terra rachada|sequeiro", "a seca e a chuva no sertão"),
    (r"donzela|\bamor\b|casa(r|mento)|noiv[ao]|\bbeijo\b|paix[ãa]o|ci[úu]me", "um amor difícil"),
    (r"dinheiro|\brico\b|riqueza|\bouro\b|\bpobre\b|pobreza|mesquinho|avaro|heran[çc]a|usur[áa]rio", "a riqueza e a pobreza"),
    (r"\bboi\b|\bvaca\b|vaqueiro|\bgado\b|curral|apart(a[çc][ãa]o|ar)|\bres\b|novilh|boiadeiro", "o vaqueiro e o gado"),
    (r"\bfome\b|farinha|feij[ãa]o|jerimum|\bcomida\b|panela|mesa vazia|passar mal", "a fome do pobre"),
    (r"viola|cantador|repente|peleja|\bglosa\b|desafio|\brima\b|\bverso\b|poeta", "a peleja dos cantadores"),
    (r"\bm[ãa]e\b|\bpai\b|\bfilho\b|\bfilha\b|[óo]rf[ãa]o|fam[íi]lia|\bber[çc]o\b", "a família e os filhos"),
    (r"\bmorte\b|caix[ãa]o|cemit[ée]rio|defunto|enterro|\bcova\b|\bluto\b|falecid", "a morte e o além"),
    (r"governo|imposto|prefeito|elei[çc][ãa]o|\bvoto\b|deputado|pol[íi]tica|coletor", "o governo e os impostos"),
    (r"navio|\bmar\b|\bonda\b|\bpeixe\b|pescador|jangada|\bmar[ée]\b", "o mar e os pescadores"),
    (r"vi[úu]v[ao]|velh[ao]|comadre|compadre|vizinh", "os casos da vizinhança"),
]

FALLBACK_THEMES = [
    "a vida no sertão", "os casos do povo", "a lida do dia a dia",
    "as histórias de antigamente", "o mundo como ele é",
]

INSTR = [
    "Faça uma estrofe de cordel sobre {t}.",
    "Escreva uns versos de cordel sobre {t}.",
    "No estilo do cordel de folheto, componha uma estrofe sobre {t}.",
    "Conte em verso de cordel: {t}.",
    "Componha uma sextilha de cordel sobre {t}.",
    "Escreva uma estrofe de folheto sobre {t}.",
]


def stanzas(path):
    text = open(path, encoding="utf-8").read()
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        joined = "\n".join(lines)
        if 3 <= len(lines) <= 12 and 40 <= len(joined) <= 700 \
           and max((len(l) for l in lines), default=0) < 90:
            out.append(joined)
    return out


def theme_for(stanza):
    low = stanza.lower()
    for pat, name in THEMES:
        if re.search(pat, low):
            return name
    return random.choice(FALLBACK_THEMES)


def main():
    files = sorted(glob.glob(os.path.join(CORDEL_DIR, "*.txt")))
    pairs, n_cont, n_stanzas = [], 0, 0

    for path in files:
        sts = stanzas(path)
        n_stanzas += len(sts)
        if len(sts) > MAX_PER_FOLHETO:                       # subsample big folhetos
            keep = sorted(random.sample(range(len(sts)), MAX_PER_FOLHETO))
        else:
            keep = list(range(len(sts)))
        for i in keep:
            s = sts[i]
            pairs.append({
                "prompt": random.choice(INSTR).format(t=theme_for(s)),
                "response": s,
            })
            if i + 1 < len(sts) and n_cont < MAX_CONTINUATION:
                pairs.append({
                    "prompt": f"Continue o cordel:\n\n{sts[i]}",
                    "response": sts[i + 1],
                })
                n_cont += 1

    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]

    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    kinds = {}
    for p in pairs:
        k = "continue" if p["prompt"].startswith("Continue") else "tema"
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(files)} folhetos, {n_stanzas} usable stanzas")
    print(f"Wrote {len(pairs)} pairs to {os.path.relpath(OUT)}  ({kinds})")


if __name__ == "__main__":
    main()
