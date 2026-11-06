"""Default acquisition paths, independent of the current working directory."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_DIR = Path(os.environ.get("TEAM_DIR", REPO_ROOT)).expanduser().resolve()
NOV2_DATA = TEAM_DIR / "1_2nov" / "nov2_data"
DATA_DIR = NOV2_DATA if NOV2_DATA.is_dir() else TEAM_DIR / "files csv"


def data_path(name):
    return str(DATA_DIR / name)
