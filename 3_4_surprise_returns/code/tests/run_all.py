"""Run every test: python code/tests/run_all.py (from the project folder)."""
import os, runpy, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
os.environ.setdefault("PROJECT_DIR", os.path.dirname(os.path.dirname(here)))
for name in ["test_surprise.py", "test_returns.py", "test_sample.py"]:
    print(f"== {name}")
    runpy.run_path(os.path.join(here, name), run_name="__main__")
print("\nall tests passed")
