# Loughran–McDonald master dictionary

The LM baseline (`code/baseline_lm.py`) needs the **Loughran–McDonald Master Dictionary** CSV.
It is free for academic use but not redistributed in this repository.

1. Download the CSV version from the Notre Dame Software Repository for Accounting and Finance:
   https://sraf.nd.edu/loughranmcdonald-master-dictionary/
2. Save it in this folder, keeping a name that contains `MasterDictionary`, e.g.
   `Loughran-McDonald_MasterDictionary_1993-2024.csv`.
3. Rerun `07_baselines.ipynb`.

The file's sha256 is part of the LM cache key (`data/text_features/lm_counts.parquet`), so a newer
dictionary is picked up automatically and every text is recounted with it. If several versions are in
this folder, the last one in alphabetical order is used. Record the version you used in
`docs/CHANGELOG.md`.

Columns read: `Word`, `Negative`, `Positive`, `Uncertainty`, `Litigious`, `Strong_Modal`,
`Weak_Modal`, `Constraining` (a positive value is the year a word was added; 0 or negative means not
in the list).
