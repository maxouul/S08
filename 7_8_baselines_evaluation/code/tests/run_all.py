"""Run every test (same entry point as part 3-4's code/tests/run_all.py)."""
import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve().parent), "-q"] + sys.argv[1:]))
