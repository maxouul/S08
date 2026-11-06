# S08 NLP

Run notebooks from the cloned repository, either from its root or their own folder. Code paths are resolved from the checkout; Google Drive is not required.

Install the dependencies in each notebook with the kernel you use to run it. Run notebooks in numerical order within each part.

Parts 1–2 use `1_2nov/nov2_data/` when present, otherwise the shared `files csv/` inputs. Parts 3–4 use `3_4_surprise_returns/cache/` and `data/`; part 5 uses `5_data cleaning/cache/` and `data/`. These private folders are not included in GitHub. Restore them from the team storage or regenerate them with the acquisition notebooks and your WRDS access.

To keep data outside the checkout, set `TEAM_DIR` to the folder containing those same subfolders before running a notebook. Code still comes from this repository. If the kernel starts outside the checkout, also set `S08_ROOT` to the repository path. Existing Drive data can be used after mounting Drive and setting `TEAM_DIR`.

Run the synthetic parts 3–4 tests from the repository root:

```sh
python 3_4_surprise_returns/code/tests/run_all.py
```

`6_Models, GPU setup and inference/` is reserved for the modeling stage and currently contains only a placeholder.
