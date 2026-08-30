"""
Tokenizers: turn text into the integer IDs the model trains on.

CharTokenizer — one ID per Unicode character. Tiny vocab (~90), but the model
    has to learn spelling from nothing and tends to emit plausible-looking
    non-words ("préibia", "digênia").

BPETokenizer — byte-level Byte-Pair Encoding, the GPT-2 approach. Start from the
    raw UTF-8 bytes of the text, then repeatedly merge the most frequent
    adjacent pair, `n_merges` times. Frequent chunks (" que", "ção", "Oliveiros")
    collapse to a single ID, so sequences are ~3x shorter (more context per
    step) and samples are built from real word-pieces instead of invented ones.

Inspired by Andrej Karpathy's minbpe, trimmed for this project.
"""

import json
import re
from collections import Counter

# Pre-tokenization (GPT-2 style): a chunk is a run of letters / digits /
# punctuation, optionally carrying ONE leading space (" que" is one chunk, so
# BPE can make it one token). Runs of other whitespace — newlines especially —
# stay on their own, so no merge can glue a line break onto a word.
_SPLIT_RE = re.compile(r" ?\d+| ?[^\d\W]+| ?[^\w\s]+|\s+")


def _pair_counts(chunks):
    counts = Counter()
    for chunk in chunks:
        counts.update(zip(chunk, chunk[1:]))
    return counts


def _merge(seq, pair, new_id):
    """Replace every adjacent occurrence of `pair` in `seq` with `new_id`."""
    out, i = [], 0
    a, b = pair
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class CharTokenizer:
    kind = "char"

    def __init__(self, chars):
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    @classmethod
    def train(cls, text, **_):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def state(self):
        return {"kind": self.kind,
                "chars": [self.itos[i] for i in range(len(self.itos))]}

    @classmethod
    def from_state(cls, state):
        return cls(state["chars"])


class BPETokenizer:
    kind = "bpe"

    def __init__(self, merges):
        # merges: [((a, b), new_id), ...] in the order they were learned
        self.merges = [(tuple(p), nid) for p, nid in merges]
        self.ranks = {pair: i for i, (pair, _) in enumerate(self.merges)}
        self.merge_id = {pair: nid for pair, nid in self.merges}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), nid in self.merges:
            self.vocab[nid] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, text, n_merges=512):
        chunks = [list(piece.encode("utf-8")) for piece in _SPLIT_RE.findall(text)]
        merges = []
        for i in range(n_merges):
            counts = _pair_counts(chunks)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break  # nothing worth merging is left
            new_id = 256 + i
            chunks = [_merge(c, pair, new_id) for c in chunks]
            merges.append((pair, new_id))
        return cls(merges)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def _encode_chunk(self, piece):
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            # merge the pair whose rule was learned earliest (lowest rank)
            pair = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.merge_id[pair])
        return ids

    def encode(self, s):
        out = []
        for piece in _SPLIT_RE.findall(s):
            out.extend(self._encode_chunk(piece))
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def state(self):
        return {"kind": self.kind,
                "merges": [[list(p), nid] for p, nid in self.merges]}

    @classmethod
    def from_state(cls, state):
        return cls([(tuple(p), nid) for p, nid in state["merges"]])


_KINDS = {"char": CharTokenizer, "bpe": BPETokenizer}


def from_state(state):
    return _KINDS[state["kind"]].from_state(state)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return from_state(json.load(f))


def save(tokenizer, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.state(), f, ensure_ascii=False)
