"""
Merge all files under data/ into a single training corpus.

Usage:
    python scripts/prepare_data.py
"""

import glob
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "corpus.txt")


def main():
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)):
        if os.path.basename(path) == "corpus.txt":
            continue  # don't fold the output back into itself
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Drop comment/placeholder lines (e.g. cordel.txt before you fill it, or the
        # attribution headers fetch_cordel.py writes). Keep blank lines: in cordel
        # they mark stanza breaks, and we want the model to learn that rhythm.
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
        chunk = "\n".join(lines).strip()
        if not chunk:
            print(f"Skipping {path} (no content yet)")
            continue
        chunks.append(chunk)
        n_real = sum(1 for ln in lines if ln.strip())
        print(f"Added {path} ({n_real} non-blank lines)")

    corpus = "\n\n".join(chunks) + "\n"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nWrote {OUTPUT_FILE} — {len(corpus):,} characters total")


if __name__ == "__main__":
    main()
