"""
Phase 3, step 1: build preference pairs for the reward model.

We have good cordel responses (data/sft*.jsonl) but no "this one is better than
that one" judgements. So we synthesise them: for each good response (`chosen`)
we make a few deliberately worse versions (`rejected`) —

  offtopic   a real stanza, but answering a DIFFERENT prompt
  shuffled   the chosen stanza with its lines in random order (breaks rhyme/flow)
  truncated  only the first two lines (incomplete)
  flattened  the stanza mashed into one prose line, lowercased (verse layout gone)

The reward model then learns: tight, on-topic, complete, laid-out cordel > the rest.
Not subtle, but it's the real mechanism (pairwise preference), and it's enough
to make best-of-N (Phase 4) work.

    python scripts/make_prefs.py          # writes data/prefs.jsonl
"""

import json
import os
import random

random.seed(1337)

BASE = os.path.dirname(__file__)
SRC = [os.path.join(BASE, "..", "data", "sft.jsonl"),
       os.path.join(BASE, "..", "data", "sft_synth.jsonl")]
OUT = os.path.join(BASE, "..", "data", "prefs.jsonl")

MAX_PAIRS = 2400


def load_pairs():
    out = []
    for p in SRC:
        if os.path.exists(p):
            out += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return out


def rejecteds(chosen, pool):
    lines = [l for l in chosen.split("\n") if l.strip()]
    variants = []

    # offtopic: someone else's stanza
    other = random.choice(pool)
    if other.strip() and other.strip() != chosen.strip():
        variants.append(("offtopic", other.strip()))

    # shuffled lines
    if len(lines) >= 4:
        sh = lines[:]
        while sh == lines:
            random.shuffle(sh)
        variants.append(("shuffled", "\n".join(sh)))

    # truncated
    if len(lines) >= 4:
        variants.append(("truncated", "\n".join(lines[:2])))

    # flattened to prose
    if len(lines) >= 3:
        variants.append(("flattened", " ".join(lines).lower()))

    return variants


def main():
    pairs = load_pairs()
    pool = [p["response"] for p in pairs]
    print(f"{len(pairs)} source responses")

    prefs = []
    for ex in pairs:
        chosen = ex["response"].strip()
        for kind, rej in rejecteds(chosen, pool):
            prefs.append({"prompt": ex["prompt"], "chosen": chosen,
                          "rejected": rej, "rejected_type": kind})

    random.shuffle(prefs)
    prefs = prefs[:MAX_PAIRS]

    with open(OUT, "w", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    by_kind = {}
    for p in prefs:
        by_kind[p["rejected_type"]] = by_kind.get(p["rejected_type"], 0) + 1
    print(f"wrote {len(prefs)} preference pairs -> {os.path.relpath(OUT)}")
    print(f"  {by_kind}")


if __name__ == "__main__":
    main()
