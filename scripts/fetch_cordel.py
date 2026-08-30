"""
Pull real public-domain cordel from Portuguese Wikisource into data/cordel/.

Wikisource hosts proofread transcriptions of folhetos by authors whose work is
in the public domain (Brazil: life + 70 years). This script fetches the parsed
HTML of a work, strips the editorial scaffolding (headers, page-number markers,
Wikidata/Commons boilerplate), keeps just the verses, and writes one plain-text
file per folheto — the format scripts/prepare_data.py already expects.

Usage:
    python scripts/fetch_cordel.py                # fetch the default folheto list
    python scripts/fetch_cordel.py "A Filha do Pescador" "A Força do Amor"

Then re-run:
    python scripts/prepare_data.py
    python scripts/train.py

Sources / attribution: pt.wikisource.org, public-domain works of
Leandro Gomes de Barros (d. 1918) and João Martins de Athayde (d. 1959).
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://pt.wikisource.org/w/api.php"
UA = "arrodeio-llm/0.1 (educational char-LM project)"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cordel")

# Wikisource page titles. All transcribed + public domain (authors d. before 1955):
# Leandro Gomes de Barros (d. 1918), Silvino Pirauá de Lima (d. 1913), and other
# folhetos in pt.wikisource's Categoria:Literatura de cordel. Add more from:
#   https://pt.wikisource.org/wiki/Categoria:Literatura_de_cordel
#   https://pt.wikisource.org/wiki/Autor:Leandro_Gomes_de_Barros
DEFAULT_WORKS = [
    # Leandro Gomes de Barros
    "A Batalha de Oliveiros com Ferrabraz",
    "O Homem que Subiu em Aeroplano até a Lua",
    "O cavalo que defecava dinheiro",
    "O Imposto de Honra",
    "A Seca do Ceará",
    "Antonio Silvino, o Rei dos Cangaceiros",
    "A peleja de Leandro Gomes com uma velha de Sergipe",
    "As Proezas de um Namorado Mofino",
    "História da Donzela Teodora (Leandro Gomes de Barros)",
    "A Vida de Pedro Cem",
    "FCRB LC6046/A Ave Maria da Eleição",
    # Silvino Pirauá de Lima and other cordelistas
    "E Tudo Vem a Ser Nada",
    "História de Crispim e Raimundo",
    "Necrológio de Francisco Romano",
    "Peleja da Alma",
    "Costumes e Usos Antigos",
    "História da Donzela Teodora (recolha de Luís da Câmara Cascudo)",
]

# lines that slip through even after we isolate the <div class="poem"> blocks
_JUNK = [
    re.compile(r"projetos\s+irm", re.I),
    re.compile(r"edi[çc][aã]o de refer", re.I),
    re.compile(r"dados de edi[çc][aã]o", re.I),
    re.compile(r"dom[ií]nio p[uú]blico", re.I),
    re.compile(r"autores brasileiros falecidos", re.I),
    re.compile(r"^\[?iniciar\]?$", re.I),
    re.compile(r"^\d{4,}[A-ZÀ-Ú]"),      # the "185830A Batalha..." metadata blob
    re.compile(r"continuar[aá] o resto", re.I),
    re.compile(r"^T\.?\s*[IVXLCDM]+\.?$"),          # volume markers: "T.VIII"
    re.compile(r"^FIM$", re.I),
]


def fetch_html(title, _tries=4):
    params = {
        "action": "parse",
        "page": title.replace(" ", "_"),
        "prop": "text",
        "redirects": "1",
        "disablelimitreport": "1",
        "disableeditsection": "1",
        "disabletoc": "1",
        "format": "json",
        "formatversion": "2",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(_tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < _tries - 1:
                wait = 5 * (attempt + 1)
                print(f"     (429 rate-limited, waiting {wait}s)")
                time.sleep(wait)
                continue
            raise
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "unknown API error"))
    return data["parse"]["text"]


def _poem_blocks(h):
    """Yield the inner HTML of every <div class="poem"> ... </div>, nesting-aware."""
    for m in re.finditer(r'<div[^>]*class="[^"]*\bpoem\b[^"]*"[^>]*>', h):
        start = m.end()
        depth = 1
        i = start
        for t in re.finditer(r"<(/?)div\b[^>]*>", h[start:]):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                i = start + t.start()
                break
        yield h[start:i]


def html_to_verses(html):
    import html as _html

    if re.search(r"página de desambiguação|índice de diferentes versões", html, re.I):
        return ""   # disambiguation / version-index page, not a text

    blocks = list(_poem_blocks(html))
    if not blocks:
        return ""   # ptwikisource verse texts are always in <div class="poem">
    h = "\n\n".join(blocks)

    h = re.sub(r"<style[^>]*>.*?</style>", "", h, flags=re.S)
    h = re.sub(r"<sup[^>]*>.*?</sup>", "", h, flags=re.S)            # footnote refs
    h = re.sub(r'<span[^>]*class="[^"]*pagenum[^"]*"[^>]*>.*?</span>', "", h, flags=re.S)
    # a <br /> and its trailing newline -> one newline; a doubled <br /> -> stanza gap
    h = re.sub(r"<br\s*/?>\s*", "\n", h)
    h = re.sub(r"</(p|div|dd|dt|li|h[1-6])>", "\n\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = _html.unescape(h)
    h = h.replace("\xa0", " ")

    h = re.sub(r"[ \t]+\n", "\n", h)
    h = re.sub(r"\n[ \t]+", "\n", h)
    h = re.sub(r"\n{3,}", "\n\n", h)

    kept = []
    for line in h.split("\n"):
        s = line.strip()
        if s and any(p.search(s) for p in _JUNK):
            continue
        kept.append(s)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

    # drop leading ALL-CAPS heading lines / blanks the poem block sometimes opens with
    lines = text.split("\n")
    while lines and (not lines[0] or (lines[0] == lines[0].upper() and len(lines[0]) > 3)):
        lines.pop(0)
    return "\n".join(lines).strip()


def slugify(title):
    s = title.lower()
    for a, b in (("àáâãä", "a"), ("éêë", "e"), ("íï", "i"), ("óôõö", "o"), ("úü", "u")):
        for ch in a:
            s = s.replace(ch, b)
    s = s.replace("ç", "c")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def main():
    works = [w.replace("_", " ") for w in sys.argv[1:]] or DEFAULT_WORKS
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for i, title in enumerate(works):
        try:
            verses = html_to_verses(fetch_html(title))
        except Exception as e:
            print(f"  !! {title}: {e}")
            continue
        if len(verses) < 400:
            print(f"  ?? {title}: {len(verses)} chars of verse — skipping "
                  f"(disambiguation page, or not fully transcribed yet)")
            continue
        path = os.path.join(OUT_DIR, slugify(title) + ".txt")
        header = f"# {title}\n# public domain — via pt.wikisource.org\n\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + verses + "\n")
        total += len(verses)
        print(f"  ok {title} -> {os.path.relpath(path)} ({len(verses):,} chars)")
        if i + 1 < len(works):
            time.sleep(3)   # be polite to the API (avoids HTTP 429)

    print(f"\nDone. {total:,} chars of cordel written to {os.path.relpath(OUT_DIR)}/")
    print("Next: python scripts/prepare_data.py && python scripts/train.py")


if __name__ == "__main__":
    main()
