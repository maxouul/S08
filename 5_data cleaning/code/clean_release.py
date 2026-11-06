"""
Step 5.2: Clean the cached raw documents into text, as a separate and versioned step.

fetch_releases.py cleaned each document while it downloaded it. Here the cleaning is a
pure function, clean(raw_bytes, filename), applied to the bytes kept by cache_raw.py, so
it can be rerun and improved without going back to EDGAR. Any change to the rules below
must bump CLEAN_VERSION, because the model-output cache is keyed on it.

Compared with to_text() in fetch_releases.py:
- the charset is read from the document instead of assuming UTF-8, and replacement
  characters are counted rather than introduced silently;
- hidden iXBRL (ix:header, display:none) is removed;
- paragraphs survive, because block tags become line breaks before the tags go;
- financial tables are kept aside in fin_tables, not thrown away. A table is financial
  if more than 30% of its characters are digits (to_text()'s rule) or if it has the
  shape of a statement: most rows carry a cell that is only a number, and no cell
  holds prose (statements with long line labels escape the digit rule);
- numeric references to cp1252 punctuation (&#146; for an apostrophe) are read as
  browsers read them, instead of becoming invisible control characters;
- standard sections (safe harbor, About X, contacts, conference call) are labelled, not
  deleted; text_body is the text without them, and is what the scorers read;
- page furniture (page numbers, "-more-", end marks such as "###") is removed, and so is
  content repeated inside a document: a paragraph pasted twice, the page header or
  footer an old filing repeats on every page, a table printed twice. The counts are
  kept, so nothing disappears unseen;
- the text is split into sentences, never across paragraphs;
- part 2's check that the document is an earnings release (2_EDGAR.ipynb: earnings
  vocabulary against that of the monthly sales and traffic reports some filers also
  file under Item 2.02) is applied to every text, as a flag (earnings_like), never as
  a deletion. Part 2's hand check found about 5% of its sample was not an earnings
  release.

Output: data/clean/clean_{year}.parquet, one row per accession, columns OUTPUT_COLUMNS
(year of the filing date). Parts 6 to 8 read accession, text_body, sha256_text,
clean_version, n_words_body and error from it.

Usage: python code/clean_release.py --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import zipfile
from multiprocessing import Pool

# v1: first version. v2: page furniture and repeated content removed (drop_repeats),
# SGML-wrapped documents read to their end, statement-shaped tables set aside,
# cp1252 numeric references decoded.
CLEAN_VERSION = "v2"

FIN_TABLE_DIGITS = 0.30          # same rule as to_text()
SEP = " "                   # paragraph mark used while the HTML is flattened

BLOCK_TAGS = ["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "table", "ul", "ol", "center", "blockquote", "title"]
SPACE_RE = re.compile(r"[ \t\r\n\f\v\xa0 -​  　]+")
META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)
HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
SGML_TEXT_END_RE = re.compile(r"</TEXT>", re.I)
SGML_TABLE_RE = re.compile(r"<TABLE>(.*?)</TABLE>", re.S | re.I)
SGML_TAG_RE = re.compile(r"</?(?:PAGE|S|C|FN|F\d+|CAPTION|PRE|DOCUMENT|TYPE|SEQUENCE|"
                         r"FILENAME|DESCRIPTION|TEXT)\b[^>]*>", re.I)
COLUMNS_RE = re.compile(r"\S {2,}\S")
PAGE_ARTEFACT_RE = re.compile(
    r"^(?:-\s*more\s*-|\(\s*more\s*\)|more|-\s*\d{1,3}\s*-|\d{1,3}"
    r"|(?:page\s+)?\d{1,3}\s+of\s+\d{1,3}|page\s+\d+|#(?:\s*#){2,}|\*(?:\s*\*){2,}"
    r"|-\s*(?:end|30)\s*-|-?\s*o\s*o\s*o\s*-?)$",
    re.I)
NUM_CELL_RE = re.compile(r"^[($€£-]*\s*\d[\d,]*(?:\.\d+)?\s*%?\)?$")
PROSE_CELL_WORDS = 40     # a table cell longer than this holds prose: a layout table
# Old documents write cp1252 punctuation as numeric references (&#146; for ’). HTML5
# reads them as cp1252, as browsers and html.unescape do; lxml leaves C1 controls.
C1_TO_CP1252 = {c: bytes([c]).decode("cp1252") for c in range(0x80, 0xA0)
                if c not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)}
REPEAT_MIN_WORDS = 8      # a repeated paragraph this long is a copy, whatever its count
REPEAT_MIN_COUNT = 3      # a shorter one goes only if it recurs, like a page header

# Part 2's earnings-release check (2_EDGAR.ipynb, automatic validation), same terms and
# rule, read on the first 8,000 characters. It only sets a flag: the text is unchanged.
EARNINGS_TERMS = ["earnings per share", "net income", "diluted", "fourth quarter",
                  "first quarter", "second quarter", "third quarter", "full year",
                  "results of operations", "financial results", "net revenue",
                  "operating income", "gaap", "guidance"]
OTHER_REPORT_TERMS = ["traffic", "revenue passenger", "load factor", "available seat miles",
                      "same-store sales for", "monthly sales", "sales for the month",
                      "unit sales for"]
EARNINGS_MIN_TERMS = 3    # earnings_like: at least this many earnings terms ...
OTHER_MAX_TERMS = 1       # ... and at most this many terms of a monthly report

# Every column of data/clean/clean_{year}.parquet, in order. A year file lacking one of
# them is rebuilt even when its clean_version is current.
OUTPUT_COLUMNS = [
    "accession", "filename", "clean_version", "error", "encoding",
    "text_full", "text_body", "sections", "sentences", "sentence_section", "fin_tables",
    "n_words", "n_words_body", "n_words_tables", "n_sentences", "n_replacement_chars",
    "n_furniture_removed", "n_repeats_removed", "n_table_copies_removed",
    "n_earnings_terms", "n_other_report_terms", "earnings_like", "sha256_text",
]

# WHATWG treats these labels as windows-1252, and so do the filers who wrote them.
CP1252_ALIASES = {"iso-8859-1", "iso8859-1", "latin-1", "latin1", "us-ascii", "ascii",
                  "windows-1252", "cp1252", "iso-8859-15"}


# --- decoding ---------------------------------------------------------------------

def decode(raw: bytes) -> tuple[str, str]:
    """Declared charset first, then strict UTF-8, then cp1252. Never replaces silently:
    whatever replacement characters remain are counted by the caller."""
    m = META_CHARSET_RE.search(raw[:8192])
    if m:
        enc = m.group(1).decode("ascii", "ignore").lower()
        enc = "cp1252" if enc in CP1252_ALIASES else enc
        if enc == "cp1252":
            # cp1252 leaves five bytes undefined; utf-8 mislabelled as latin-1 is common.
            try:
                return raw.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                pass
        try:
            return raw.decode(enc), enc
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    return raw.decode("cp1252", errors="replace"), "cp1252"


# --- tables -----------------------------------------------------------------------

def is_financial(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    return digits > FIN_TABLE_DIGITS * max(letters + digits, 1)


def is_financial_table(txt: str) -> bool:
    """The digit rule of to_text(), or the shape of a financial statement: at least
    three rows, and half of them, carry a cell that is only a number, while no cell
    holds prose. Statements with long line labels ("Selling, general and administrative
    expenses") fall under 30% digits but are tables all the same; a layout table
    wrapping paragraphs has long cells and stays text."""
    if is_financial(txt):
        return True
    rows = [r.split("\t") for r in txt.split("\n") if r.strip()]
    if any(len(c.split()) > PROSE_CELL_WORDS for r in rows for c in r):
        return False
    numeric = sum(any(NUM_CELL_RE.match(c.strip()) for c in r) for r in rows)
    return numeric >= 3 and numeric >= 0.5 * len(rows)


def html_table_text(table) -> str:
    """One line per row, cells separated by tabs."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [SPACE_RE.sub(" ", td.get_text(" ")).strip()
                 for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)


# --- plain text (old .txt filings and <PRE> blocks) --------------------------------

def plain_paragraphs(text: str, fin_tables: list[str]) -> list[str]:
    """Blank lines separate paragraphs; single newlines are hard wraps. A block laid
    out in columns that is mostly digits is a financial table and goes to fin_tables."""

    def keep_table(m: re.Match) -> str:
        body = SGML_TAG_RE.sub("", m.group(1))
        if is_financial(body):
            fin_tables.append(body.strip("\n"))
            return "\n\n"
        return "\n\n" + body + "\n\n"

    text = SGML_TABLE_RE.sub(keep_table, text)
    text = SGML_TAG_RE.sub("", text)
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        columnar = sum(bool(COLUMNS_RE.search(ln.strip())) for ln in lines)
        if len(lines) >= 2 and columnar >= len(lines) / 2 and is_financial(block):
            fin_tables.append("\n".join(ln.rstrip() for ln in lines))
            continue
        para = SPACE_RE.sub(" ", " ".join(lines)).strip()
        if para:
            out.append(para)
    return out


# --- HTML -------------------------------------------------------------------------

def html_paragraphs(doc: str, fin_tables: list[str]) -> list[str]:
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(doc, "lxml")
    except Exception:
        soup = BeautifulSoup(doc, "html.parser")

    for t in soup.find_all(["script", "style", "head", "noscript"]):
        t.decompose()
    # Inline XBRL hides its facts in ix:header; some filers hide whole blocks.
    for t in soup.find_all(lambda tag: tag.name in ("ix:header", "ix:hidden")):
        t.decompose()
    for t in soup.find_all(style=HIDDEN_RE):
        t.decompose()

    # Innermost tables first: reversed document order visits children before parents.
    for table in reversed(soup.find_all("table")):
        if table.parent is None:          # removed with an enclosing table already
            continue
        txt = html_table_text(table)
        if txt and is_financial_table(txt):
            fin_tables.append(txt)
            table.replace_with(SEP)
        # a layout table stays; its rows become paragraphs below

    for pre in soup.find_all(["pre", "xmp", "plaintext"]):
        pre.replace_with(SEP + SEP.join(plain_paragraphs(pre.get_text(), fin_tables)) + SEP)

    for br in soup.find_all("br"):
        br.replace_with(SEP)
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before(SEP)
        tag.insert_after(SEP)
    for td in soup.find_all(["td", "th"]):
        td.insert_after(" ")

    text = SPACE_RE.sub(" ", soup.get_text().translate(C1_TO_CP1252))
    return [p.strip() for p in text.split(SEP) if p.strip()]


def is_html(raw_head: str, filename: str) -> bool:
    if filename.lower().endswith((".htm", ".html", ".xhtml")):
        return True
    # old .txt filings carry SGML <TABLE> and <PAGE> tags, so only these count as HTML
    return bool(re.search(r"<(html|body|p|div|font)\b", raw_head, re.I))


# --- sections ---------------------------------------------------------------------

HEAD_PATTERNS = [
    ("safe_harbor", re.compile(r"forward[- ]looking|safe harbor|cautionary (statement|note)"
                               r"|private securities litigation", re.I)),
    ("about", re.compile(r"^about\s+\S", re.I)),
    ("contacts", re.compile(r"^(investor|media|press|analyst|company|financial)?\s*"
                            r"(relations\s*)?(and media\s*)?contacts?\b"
                            r"|^for (further|more|additional) information", re.I)),
    ("conference_call", re.compile(r"conference call|webcast|earnings call", re.I)),
]
SAFE_HARBOR_PARA = re.compile(r"forward[- ]looking (statement|information)", re.I)
RISK_WORDS = re.compile(r"\b(risks?|uncertaint)", re.I)
CALL_PARA = re.compile(r"\b(conference call|webcast|earnings call)\b", re.I)
CALL_DETAIL = re.compile(r"\b(dial|replay|listen|passcode|pass code|toll[- ]free|"
                         r"conference id|www\.|https?://)", re.I)
CONTACT_LINE = re.compile(r"@|\(\d{3}\)|\d{3}[-.\s]\d{3}[-.\s]\d{4}", re.I)


def is_heading(p: str) -> bool:
    words = p.split()
    if 0 < len(words) <= 12 and not p.rstrip().endswith((".", ";", ",")):
        return True
    # "About Acme Corp." ends with the period of the abbreviation
    return 0 < len(words) <= 8 and any(rx.search(p) for _, rx in HEAD_PATTERNS)


def label_paragraphs(paras: list[str]) -> list[str]:
    """Heading-driven labelling. A heading that matches a pattern opens that section;
    any other heading closes it, except inside contacts, where names and phone numbers
    look like headings and only a long paragraph closes the block. Without a heading,
    a forward-looking paragraph that cites risks opens the safe harbor, and a
    paragraph giving dial-in details is labelled conference_call on its own."""
    labels, current = [], "body"
    for p in paras:
        lab = None
        if is_heading(p):
            for name, rx in HEAD_PATTERNS:
                if rx.search(p):
                    lab = name
                    break
            if lab:
                current = lab
            elif current == "contacts" and (CONTACT_LINE.search(p) or len(p.split()) <= 6):
                pass
            else:
                current = "body"
            labels.append(current)
            continue
        if current == "contacts" and len(p.split()) > 25:
            current = "body"
        if SAFE_HARBOR_PARA.search(p) and RISK_WORDS.search(p):
            current = "safe_harbor"
        elif current == "body" and CALL_PARA.search(p) and CALL_DETAIL.search(p) \
                and len(p.split()) < 200:
            labels.append("conference_call")
            continue
        labels.append(current)
    return labels


# --- sentences --------------------------------------------------------------------

FIN_ABBREV = {"inc", "corp", "co", "ltd", "llc", "plc", "lp", "llp", "n.a", "s.a", "n.v",
              "a.g", "u.s", "u.k", "no", "nos", "vs", "approx", "est", "e.g", "i.e",
              "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
              "nov", "dec", "mr", "mrs", "ms", "dr", "jr", "sr", "st", "ave", "dept",
              "fig", "yr", "yrs", "mo", "qtr", "avg", "pct", "bil", "mil", "ft", "sq",
              "intl", "assn", "bros", "cos", "hldgs", "mfg", "svcs", "tel", "p.m", "a.m"}
_SPLITTER = None


def sentence_splitter():
    """NLTK punkt for English, with financial abbreviations added so that 'Inc.' or
    'U.S.' do not end a sentence. '$1.2 million' and 'Q3' never split under punkt."""
    global _SPLITTER
    if _SPLITTER is None:
        import nltk
        try:
            from nltk.tokenize import PunktTokenizer          # nltk >= 3.9
            try:
                tok = PunktTokenizer("english")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                tok = PunktTokenizer("english")
        except ImportError:                                   # nltk < 3.9
            try:
                tok = nltk.data.load("tokenizers/punkt/english.pickle")
            except LookupError:
                nltk.download("punkt", quiet=True)
                tok = nltk.data.load("tokenizers/punkt/english.pickle")
        tok._params.abbrev_types.update(FIN_ABBREV)
        _SPLITTER = tok
    return _SPLITTER


# --- repeated content -------------------------------------------------------------

def drop_repeats(paras: list[str]) -> tuple[list[str], int, int]:
    """Remove page furniture, and every later copy of a paragraph already seen in the
    document. A short paragraph (a heading such as "Highlights") is dropped only if it
    recurs at least REPEAT_MIN_COUNT times, as page headers and footers do: a heading
    met twice may open a real section, and the labelling relies on it.
    Returns the kept paragraphs, the furniture removed and the copies removed."""
    from collections import Counter
    keys = [SPACE_RE.sub(" ", p).strip().lower() for p in paras]
    counts = Counter(keys)
    seen: set[str] = set()
    out, n_furniture, n_copies = [], 0, 0
    for p, key in zip(paras, keys):
        # furniture: page numbers and end marks, or a line without a single letter
        # (a stray table row such as "$ 19 $ 14"), which cannot be prose
        if PAGE_ARTEFACT_RE.match(key) or not any(ch.isalpha() for ch in key):
            n_furniture += 1
            continue
        if key in seen and (len(key.split()) >= REPEAT_MIN_WORDS
                            or counts[key] >= REPEAT_MIN_COUNT):
            n_copies += 1
            continue
        seen.add(key)
        out.append(p)
    return out, n_furniture, n_copies


# --- the pure function ------------------------------------------------------------

def clean(raw_bytes: bytes, filename: str) -> dict:
    text, encoding = decode(raw_bytes)
    fin_tables: list[str] = []

    # EDGAR's SGML wrapper: the document is what lies between <TEXT> and </TEXT>. The
    # opening tag sits in the header, but the closing one can be anywhere in the file.
    start = text[:20000].upper().find("<TEXT>")
    if start >= 0:
        end = SGML_TEXT_END_RE.search(text, start)
        inner = text[start + len("<TEXT>"):end.start() if end else len(text)]
    else:
        inner = text
    if is_html(inner[:5000], filename):
        paras = html_paragraphs(inner, fin_tables)
    else:
        paras = plain_paragraphs(inner, fin_tables)

    paras, n_furniture, n_copies = drop_repeats(paras)
    n_tables_found = len(fin_tables)
    fin_tables = list(dict.fromkeys(fin_tables))        # a table printed twice, once

    labels = label_paragraphs(paras)
    text_full = "\n".join(paras)

    sections, pos = [], 0
    for p, lab in zip(paras, labels):
        start, end = pos, pos + len(p)
        if sections and sections[-1]["label"] == lab:
            sections[-1]["end"] = end
        else:
            sections.append({"label": lab, "start": start, "end": end})
        pos = end + 1
    text_body = "\n".join(p for p, lab in zip(paras, labels) if lab == "body")

    splitter = sentence_splitter()
    sentences, sentence_section = [], []
    for p, lab in zip(paras, labels):
        for s in splitter.tokenize(p):
            sentences.append(s)
            sentence_section.append(lab)

    head = text_full[:8000].lower()
    n_earnings = sum(t in head for t in EARNINGS_TERMS)
    n_other = sum(t in head for t in OTHER_REPORT_TERMS)

    return {
        "clean_version": CLEAN_VERSION,
        "encoding": encoding,
        "text_full": text_full,
        "text_body": text_body,
        "sections": sections,
        "sentences": sentences,
        "sentence_section": sentence_section,
        "fin_tables": fin_tables,
        "n_words": len(text_full.split()),
        "n_words_body": len(text_body.split()),
        "n_words_tables": sum(len(t.split()) for t in fin_tables),
        "n_sentences": len(sentences),
        "n_replacement_chars": text.count("�"),
        "n_furniture_removed": n_furniture,
        "n_repeats_removed": n_copies,
        "n_table_copies_removed": n_tables_found - len(fin_tables),
        "n_earnings_terms": n_earnings,
        "n_other_report_terms": n_other,
        "earnings_like": n_earnings >= EARNINGS_MIN_TERMS and n_other <= OTHER_MAX_TERMS,
        "sha256_text": hashlib.sha256(text_full.encode("utf-8")).hexdigest(),
    }


# --- driver -----------------------------------------------------------------------

def _work(args: tuple[str, str]) -> dict:
    zip_path, member = args
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member)
    acc, filename = member.split("__", 1)
    try:
        rec = clean(raw, filename)
        rec["error"] = ""
    except Exception as e:                 # counted and reported, never dropped quietly
        rec = {"clean_version": CLEAN_VERSION, "error": f"{type(e).__name__}: {e}"[:300]}
    rec["accession"] = acc
    rec["filename"] = filename
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None, help="default: cache/raw")
    ap.add_argument("--out-dir", default=None, help="default: data/clean")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--years", default="", help="e.g. 2006-2010; default all")
    ap.add_argument("--force", action="store_true",
                    help="rebuild years already cleaned with this CLEAN_VERSION")
    args = ap.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq

    import settings as S

    args.raw_dir = args.raw_dir or str(S.RAW_DIR)
    args.out_dir = args.out_dir or str(S.CLEAN_DIR)
    os.makedirs(args.out_dir, exist_ok=True)
    zips = sorted(f for f in os.listdir(args.raw_dir) if re.fullmatch(r"raw_\d{4}\.zip", f))
    if args.years:
        lo, hi = (int(x) for x in args.years.split("-"))
        zips = [f for f in zips if lo <= int(f[4:8]) <= hi]
    if not zips:
        print(f"no raw_YYYY.zip in {args.raw_dir}")
        return 1

    sentence_splitter()                    # download punkt once, before forking
    total = errors = 0
    for name in zips:
        year = name[4:8]
        out = os.path.join(args.out_dir, f"clean_{year}.parquet")
        if os.path.exists(out) and not args.force:
            v = pq.read_table(out, columns=["clean_version"]).column(0).unique().to_pylist()
            if v == [CLEAN_VERSION] and set(OUTPUT_COLUMNS) <= set(pq.read_schema(out).names):
                print(f"  {year}: already cleaned with {CLEAN_VERSION}, skipped")
                continue
        zp = os.path.join(args.raw_dir, name)
        with zipfile.ZipFile(zp) as z:
            members = [m for m in z.namelist() if "__" in m and not m.endswith("__index.htm")]
        with Pool(args.workers) as pool:
            recs = pool.map(_work, [(zp, m) for m in members], chunksize=32)
        df = pd.DataFrame(recs).reindex(columns=OUTPUT_COLUMNS)
        bad = int((df["error"] != "").sum())
        tmp = out + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)               # a crash never leaves a half-written year
        total += len(df)
        errors += bad
        print(f"  {year}: {len(df):,} cleaned, {bad:,} errors", flush=True)

    print(f"\ndocuments cleaned : {total:,}")
    print(f"errors            : {errors:,}")
    print(f"clean version     : {CLEAN_VERSION}")
    print(f"written           : {args.out_dir}/clean_YYYY.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
