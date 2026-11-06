"""
Saving results automatically (task 8: "save all regression outputs and tables
automatically").

Every table is written three ways, because it has three readers:
    .csv  full precision, for the teammates who merge or re-plot it
    .md   rounded, readable in VS Code / GitHub
    .tex  rounded, \\input{} directly into the report
Regression fits are saved as statsmodels' full text summary (.txt).

Every save is appended to logs/runs.jsonl with the time, the file, its sha256,
the git commit of this repository (if any) and the versions of the upstream
inputs (surprise_version, returns_version, sample_version, clean_version), so a
number in the report can be traced to the exact code and data that produced it.

Only summary statistics may be saved under results/: never row-level WRDS data
(the part 3-4 README explains the licence). ``save_table`` refuses a table with
an event_id/permno/accession column for that reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

ROW_LEVEL_KEYS = {"event_id", "permno", "accession", "cik", "ticker"}
_CONTEXT: dict = {}


def set_context(**kw) -> None:
    """Upstream versions etc. recorded with every saved file of this run."""
    _CONTEXT.update(kw)


_CODE_DIR = Path(__file__).resolve().parent
_GIT_CACHE: dict = {}


def _git_commit() -> str | None:
    """
    Commit of the repository that holds *this code* (not of STEP78_DIR: on Drive or
    with STEP78_DIR overridden, the output folder is often not a git checkout, which
    is why early logs recorded "git": null). "-dirty" is appended when the parts 7-8
    code differs from that commit. Falls back to $GIT_COMMIT (e.g. set on Colab).
    """
    if "v" in _GIT_CACHE:
        return _GIT_CACHE["v"]
    v = None
    for d in (_CODE_DIR, S.STEP78_DIR):
        try:
            h = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            if h:
                dirty = subprocess.run(["git", "-C", str(d), "status", "--porcelain", "--",
                                        str(_CODE_DIR)],
                                       capture_output=True, text=True, timeout=5).stdout.strip()
                v = h + ("-dirty" if dirty else "")
                break
        except Exception:
            continue
    v = v or os.environ.get("GIT_COMMIT") or None
    _GIT_CACHE["v"] = v
    return v


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _log(path: Path, kind: str, **extra) -> None:
    S.LOGS.mkdir(parents=True, exist_ok=True)
    rec = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
           "path": str(path.relative_to(S.STEP78_DIR)) if path.is_relative_to(S.STEP78_DIR) else str(path),
           "sha256": _sha(path), "git": _git_commit(), **_CONTEXT, **extra}
    with open(S.RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _md(df: pd.DataFrame, index: bool) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v)
                                       for v in r.to_numpy()) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str, caption: str = "", index: bool = False,
               digits: int = 4, folder: Path | None = None) -> Path:
    """Write results/tables/<name>.{csv,md,tex} and log it."""
    leaked = ROW_LEVEL_KEYS & set(map(str, df.columns)) | ROW_LEVEL_KEYS & set(map(str, df.index.names))
    if leaked:
        raise ValueError(f"{name}: columns {leaked} look row-level; results/ holds summary statistics only")
    folder = folder or S.TABLES
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / f"{name}.csv"
    df.to_csv(csv, index=index)
    rounded = df.copy()
    for c in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[c]):
            rounded[c] = rounded[c].round(digits)
    head = f"<!-- {caption} -->\n" if caption else ""
    (folder / f"{name}.md").write_text(head + _md(rounded, index), encoding="utf-8")
    try:
        tex = rounded.to_latex(index=index, caption=caption or None, label=f"tab:{name}",
                               escape=True, na_rep="")
        (folder / f"{name}.tex").write_text(tex, encoding="utf-8")
    except Exception as e:                      # jinja2 missing: csv/md are enough
        print(f"  (no .tex for {name}: {e})")
    _log(csv, "table", name=name, rows=len(df), caption=caption)
    return csv


def save_fit(res, name: str, note: str = "") -> Path:
    """statsmodels results -> results/regressions/<name>.txt"""
    S.REGRESSIONS.mkdir(parents=True, exist_ok=True)
    p = S.REGRESSIONS / f"{name}.txt"
    body = (note + "\n\n" if note else "") + str(res.summary())
    p.write_text(body, encoding="utf-8")
    _log(p, "regression", name=name, nobs=int(res.nobs))
    return p


def save_figure(fig, name: str) -> Path:
    S.FIGURES.mkdir(parents=True, exist_ok=True)
    p = S.FIGURES / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    _log(p, "figure", name=name)
    return p


def upstream_versions(events: pd.DataFrame, texts: pd.DataFrame | None = None) -> dict:
    """The version stamps of the inputs, for set_context()."""
    v = {c: ",".join(sorted(events[c].dropna().astype(str).unique()))
         for c in ["surprise_version", "returns_version", "sample_version"] if c in events}
    if texts is not None and "clean_version" in texts:
        v["clean_version"] = ",".join(sorted(texts["clean_version"].dropna().astype(str).unique()))
    return v
