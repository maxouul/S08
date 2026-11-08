"""where the files live.

1_2nov/data  the base files, what we pulled from WRDS and the SEC.
             nothing writes here. (this folder used to be called nov2_data)
files csv    everything we build, including rebuilt versions of a base file.

a script reads the version in files csv if there is one, the base otherwise.
that is what breaks the loop: build_event_table.py used to write back on top
of the file fetch_acceptance.py had just read, and nothing downstream could
tell which version it got.

    from paths import src, dst
    ev = pd.read_csv(src("event_table.csv"))
    ev.to_csv(dst("event_table.csv"), index=False)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# the defaults come from scripts/project_paths.py, so they do not depend on
# the folder a script is started from. both can still be pointed somewhere
# else from the environment, which is what the notebook does.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import DATA_DIR, PART1_DIR

BASE = os.environ.get("NOV2_BASE", str(PART1_DIR / "data"))
BUILT = os.environ.get("NOV2_BUILT", str(DATA_DIR))


def src(name: str) -> str:
    """the file to read: the built version if we have one, else the base."""
    built = os.path.join(BUILT, name)
    if os.path.exists(built) and os.path.getsize(built) > 0:
        return built
    base = os.path.join(BASE, name)
    if os.path.exists(base):
        return base
    # not built yet and not a base file. return the built path so the error
    # names the folder the file is supposed to appear in.
    return built


def dst(name: str) -> str:
    """the file to write. always files csv, never the base folder."""
    os.makedirs(BUILT, exist_ok=True)
    return os.path.join(BUILT, name)


def is_built(name: str) -> bool:
    """did a script already rebuild this one?"""
    p = os.path.join(BUILT, name)
    return os.path.exists(p) and os.path.getsize(p) > 0


def where(name: str) -> str:
    """for printing: "base" or "built", so a run says what it read."""
    return "built" if is_built(name) else "base"
