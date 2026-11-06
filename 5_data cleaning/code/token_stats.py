"""
Step 5.4: Length of every text_body in tokens for each candidate model, and the pieces
it is cut into to fit the model's window. The pieces are cached: they are the
tokenized inputs the scoring step reads.

Each model is counted with its own tokenizer, because words are a poor proxy (FinBERT's
WordPiece and Llama's BPE differ by a third on numbers and tickers):
- FinBERT, ProsusAI/finbert: 512 tokens, 510 left after [CLS] and [SEP];
- Llama 3.1 8B Instruct: gated, needs HF_TOKEN in the environment; skipped otherwise;
- ChronoGPT, manelalab/chrono-gpt-v1-20231231: the GPT-2 tokenizer through tiktoken,
  1,792-token context. The 2023 vintage matches Llama 3.1's December 2023 cutoff;
  the model cards' snippets load other vintages, so check this id before scoring.

chunk() cuts the body sentences (those labelled "body" by clean_release.py) at sentence
boundaries, with an overlap of whole sentences. No sentence is split; one longer than
the window becomes its own piece, flagged truncated, and the scorer cuts it to the
window. The window left for the text is a choice (MODELS below), not the raw context
length: the prompt and the answer need room too.

The cache of pieces, data/chunks/<model>/chunks_<year>.parquet, has one row per piece:
accession, chunk_idx, start and end (indices into the body sentences, end exclusive),
n_tokens, truncated, and tokens_text, the token count of the whole text_body. Every row
also carries what the pieces depend on: sha256_text, clean_version, tokenizer_id,
max_tokens and overlap. A text with no body sentence gets one row with chunk_idx -1.
Token ids themselves are not stored: they are rebuilt exactly, and fast, from the
boundaries (see pieces()), while storing them would take gigabytes.
A year whose cache already matches the cleaned texts and the settings is skipped.

Output: the cache above; data/token_stats.parquet, one row per accession (tokens,
pieces and truncated sentences per model), rebuilt from the cache; and
results/token_quantiles.csv (quantiles by year and model).

In a scoring script:
    from token_stats import pieces
    rows = chunks[chunks.accession == acc]          # from data/chunks/<model>/
    texts = pieces(clean_row, rows)                 # the text of each piece, in order

Usage: python code/token_stats.py --overlap 1
       python code/token_stats.py --models finbert,chronogpt
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import settings as S

MODELS = {
    # name: (tokenizer id, tokens left for the text in one window)
    "finbert": ("ProsusAI/finbert", 510),
    "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", 8000),
    "chronogpt": ("manelalab/chrono-gpt-v1-20231231", 1500),
}
BATCH = 500        # texts tokenized together; bounds memory on Colab
PARAMS = ["sha256_text", "clean_version", "tokenizer_id", "max_tokens", "overlap"]


def load_tokenizer(name: str):
    """A function list[str] -> list[int] of token counts, or None if unavailable."""
    tok_id = MODELS[name][0]
    if name == "chronogpt":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda texts: [len(t) for t in enc.encode_ordinary_batch(texts)]
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(tok_id, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        print(f"  {name}: tokenizer unavailable ({type(e).__name__}); "
              f"set HF_TOKEN for gated models", flush=True)
        return None
    return lambda texts: [len(x) for x in
                          tok(texts, add_special_tokens=False)["input_ids"]] if texts else []


def chunk_lens(lens: list[int], max_tokens: int, overlap_sentences: int = 1) -> list[dict]:
    """chunk() on sentence lengths already counted."""
    out, i, n = [], 0, len(lens)
    while i < n:
        j, tot = i, 0
        while j < n and tot + lens[j] <= max_tokens:
            tot += lens[j]
            j += 1
        if j == i:                        # one sentence longer than the window
            out.append({"start": i, "end": i + 1, "n_tokens": max_tokens, "truncated": True})
            j = i + 1
        else:
            out.append({"start": i, "end": j, "n_tokens": tot, "truncated": False})
        if j >= n:
            break
        # drop overlap sentences until the next sentence fits, so every piece
        # reaches past the end of the previous one
        i = max(j - overlap_sentences, i + 1)
        while i < j and sum(lens[i:j]) + lens[j] > max_tokens:
            i += 1
    return out


def chunk(sentences: list[str], tokenizer, max_tokens: int,
          overlap_sentences: int = 1) -> list[dict]:
    """Pieces of at most max_tokens, cut between sentences, each starting with the last
    overlap_sentences sentences of the previous one. Returns [{start, end, n_tokens,
    truncated}], end exclusive, in sentence indices. Token counts are summed sentence
    by sentence, which can differ by a token or two from encoding the joined text."""
    if not len(sentences):
        return []
    return chunk_lens(tokenizer(list(sentences)), max_tokens, overlap_sentences)


def body_sentences(clean_row) -> list[str]:
    """The sentences of text_body, in order: the list chunk indices refer to."""
    return [s for s, lab in zip(clean_row["sentences"], clean_row["sentence_section"])
            if lab == "body"]


def pieces(clean_row, chunk_rows: pd.DataFrame) -> list[str]:
    """The text of each cached piece of one document, in order. clean_row is its row of
    data/clean/clean_<year>.parquet; chunk_rows its rows of the chunk cache. Refuses
    pieces cut from another version of the text."""
    rows = chunk_rows[chunk_rows.chunk_idx >= 0].sort_values("chunk_idx")
    if len(rows) and (rows.sha256_text != clean_row["sha256_text"]).any():
        raise ValueError(f"{clean_row['accession']}: pieces were cut from another text; "
                         f"rerun token_stats.py")
    sents = body_sentences(clean_row)
    return [" ".join(sents[a:b]) for a, b in zip(rows.start, rows.end)]


def cut_year(df: pd.DataFrame, name: str, count, overlap: int) -> pd.DataFrame:
    """All pieces of one year's texts for one model, in batches."""
    tok_id, max_tokens = MODELS[name]
    out = []
    for lo in range(0, len(df), BATCH):
        part = df.iloc[lo:lo + BATCH]
        sents = [body_sentences(r) for _, r in part.iterrows()]
        flat = count([s for ss in sents for s in ss])
        totals = count(part.text_body.fillna("").tolist())
        k = 0
        for (_, r), ss, total in zip(part.iterrows(), sents, totals):
            lens, k = flat[k:k + len(ss)], k + len(ss)
            cuts = chunk_lens(lens, max_tokens, overlap) or \
                [{"start": 0, "end": 0, "n_tokens": 0, "truncated": False}]
            for idx, c in enumerate(cuts):
                out.append({"accession": r.accession,
                            "chunk_idx": idx if lens else -1, **c, "tokens_text": total,
                            "sha256_text": r.sha256_text, "clean_version": r.clean_version,
                            "tokenizer_id": tok_id, "max_tokens": max_tokens,
                            "overlap": overlap})
    return pd.DataFrame(out)


def up_to_date(path: str, df: pd.DataFrame, name: str, overlap: int) -> bool:
    if not os.path.exists(path):
        return False
    old = pd.read_parquet(path, columns=["accession"] + PARAMS).drop_duplicates("accession")
    tok_id, max_tokens = MODELS[name]
    same_settings = (old.tokenizer_id.eq(tok_id).all() and old.max_tokens.eq(max_tokens).all()
                     and old.overlap.eq(overlap).all())
    texts = set(zip(df.accession, df.sha256_text))
    return same_settings and set(zip(old.accession, old.sha256_text)) == texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(S.CLEAN_DIR))
    ap.add_argument("--chunk-dir", default=str(S.CHUNK_DIR))
    ap.add_argument("--out", default=str(S.TOKEN_STATS))
    ap.add_argument("--quantiles", default=str(S.TOKEN_QUANTILES))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--overlap", type=int, default=1)
    args = ap.parse_args()

    years = sorted(f for f in os.listdir(args.clean_dir)
                   if f.startswith("clean_") and f.endswith(".parquet"))
    if not years:
        print(f"no clean_YYYY.parquet in {args.clean_dir}: run clean_release.py first")
        return 1

    done_models = []
    for name in args.models.split(","):
        count = None
        mdir = os.path.join(args.chunk_dir, name)
        os.makedirs(mdir, exist_ok=True)
        for f in years:
            year = f[6:10]
            df = pd.read_parquet(os.path.join(args.clean_dir, f),
                                 columns=["accession", "clean_version", "sha256_text",
                                          "text_body", "sentences", "sentence_section"])
            df = df[df.sha256_text.notna()]
            path = os.path.join(mdir, f"chunks_{year}.parquet")
            if up_to_date(path, df, name, args.overlap):
                print(f"  {name} {year}: cache up to date, skipped", flush=True)
                continue
            if count is None:
                count = load_tokenizer(name)
                if count is None:
                    break
            cuts = cut_year(df, name, count, args.overlap)
            cuts.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)          # never a half-written year
            print(f"  {name} {year}: {len(df):,} texts, "
                  f"{int((cuts.chunk_idx >= 0).sum()):,} pieces", flush=True)
        else:
            done_models.append(name)

    # per-text table, rebuilt from the cache of every model that is complete
    stats = None
    for name in done_models:
        c = pd.concat([pd.read_parquet(os.path.join(args.chunk_dir, name,
                                                    f"chunks_{f[6:10]}.parquet"))
                       .assign(year=int(f[6:10])) for f in years], ignore_index=True)
        g = c.groupby(["accession", "year"])
        s = pd.DataFrame({f"tokens_{name}": g.tokens_text.first(),
                          f"chunks_{name}": g.chunk_idx.apply(lambda x: int((x >= 0).sum())),
                          f"truncated_{name}": g.truncated.sum().astype(int)}).reset_index()
        stats = s if stats is None else stats.merge(s, on=["accession", "year"], how="outer")
    if stats is None:
        print("no model could be run")
        return 1
    os.makedirs(os.path.dirname(args.quantiles), exist_ok=True)
    stats.to_parquet(args.out, index=False)

    qs = [0.1, 0.5, 0.9, 0.99]
    rows = []
    for name in done_models:
        for year, g in stats.groupby("year"):
            q = g[f"tokens_{name}"].quantile(qs)
            rows.append({"model": name, "year": year, "n": len(g),
                         **{f"tokens_p{int(100 * k)}": int(v) for k, v in q.items()},
                         "chunks_median": float(g[f"chunks_{name}"].median()),
                         "share_one_chunk": round(float((g[f"chunks_{name}"] == 1).mean()), 4),
                         "truncated_sentences": int(g[f"truncated_{name}"].sum())})
    pd.DataFrame(rows).to_csv(args.quantiles, index=False)

    print(f"\ntexts: {len(stats):,}")
    for name in done_models:
        s = stats[f"tokens_{name}"]
        print(f"  {name:<11} median {int(s.median()):>6,}  p90 {int(s.quantile(.9)):>6,}  "
              f"one piece {100 * (stats[f'chunks_{name}'] == 1).mean():5.1f}%  "
              f"truncated sentences {int(stats[f'truncated_{name}'].sum()):,}")
    skipped = [m for m in args.models.split(",") if m not in done_models]
    if skipped:
        print(f"not run: {', '.join(skipped)}")
    print(f"-> {args.chunk_dir}/<model>/\n-> {args.out}\n-> {args.quantiles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
