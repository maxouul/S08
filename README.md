# S08 NLP

Open the notebooks in Jupyter or VS Code from this checkout. Paths are defined in `scripts/project_paths.py` and work from the repository root or a notebook subfolder. Google Colab and Google Drive are no longer required. `%pip` cells install into the active notebook kernel; Python subprocesses use that same interpreter.

Every part uses the same layout: `code/` (Python modules), `data/` (inputs and built tables), `cache/` (downloads), `results/` (tables and figures for the report) and `logs/`.

| Location inside S08 | Use |
| --- | --- |
| `scripts/project_paths.py` | Shared path configuration for every part |
| `files csv/` | Shared CSV files built by part 1 and read by parts 2–5 |
| `1_2nov/code/` | Part 1 pipeline scripts (written by the `%%writefile` cells of the notebook) |
| `1_2nov/data/` | Part 1 base files pulled from WRDS and the SEC (formerly `nov2_data`) |
| `1_2nov/cache/` | Downloads made by the part 1 scripts (SEC bulk archive, release zips) |
| `1_2nov/results/`, `2_EDGAR text/results/` | Figures and reports for parts 1–2 |
| `2_EDGAR text/Manual validation/` | Extraction and timing hand-check sheets |
| `3_4_surprise_returns/cache/`, `data/` | Private financial inputs and built tables |
| `3_4_surprise_returns/results/` | Summary tables for parts 3–4 |
| `5_data cleaning/cache/`, `data/`, `results/` | Cleaning caches, outputs and summaries |
| `6_Models, GPU setup and inference/` | Modeling stage (currently a placeholder) |

Run notebooks in numerical order within each part. Private caches and datasets belong in the stage folders and remain ignored by Git. A fresh clone needs those folders from team storage, or regeneration using the acquisition notebooks and WRDS access. No credentials are embedded: set `HF_TOKEN` in your environment before launching Jupyter when using gated Hugging Face models.

Defaults keep inputs and outputs inside S08. Advanced overrides are optional: `TEAM_DIR` relocates the data tree, `PROJECT_DIR` relocates parts 3–4, and `STEP5_DIR` relocates part 5. Code always comes from the checkout. If the notebook kernel starts outside S08, set `S08_ROOT` to the checkout path before running the setup cell.

Run checks from the repository root with your Python interpreter:

```sh
python -m unittest discover -s tests
python 3_4_surprise_returns/code/tests/run_all.py
python "5_data cleaning/code/test_clean_release.py"
```

The cleaning tests require the dependencies installed by the part 5 notebook. Path tests need only the Python standard library; the parts 3–4 tests use synthetic data.
