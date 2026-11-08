"""
Baseline 3 (task 7): TF-IDF features with a penalised linear model.

Why this baseline
-----------------
The LM dictionary is fixed; TF-IDF *learns* which words matter from past
releases and returns. It is the strongest "simple" text model: if an 8-billion-
parameter LLM cannot beat a ridge regression on word counts trained on earlier
years, the LLM's apparent skill is not about understanding the text (and may be
memorisation, task 9).

Specification (and why)
-----------------------
* Text: part 5's ``text_body`` with numbers, amounts, percentages and years
  replaced by placeholders (text_utils.mask_numbers).
* Features: unigrams and bigrams (bigrams catch "not meet", "below expectations",
  "raises guidance"), sublinear tf (log counts, so a long release that repeats
  "growth" 40 times is not 40x more positive), l2-normalised rows (length-free),
  document frequency between ``min_df`` documents and 50% of documents, at most
  ``max_features`` terms.
* **Firm filter**: a term must occur in releases of at least ``min_firms``
  distinct firms in the training window. Without it the model learns product and
  executive names ("iphone", "mr smith") that identify the firm and its past
  returns rather than the news; that is a form of memorisation we do not want in
  a baseline.
* Models:
    ``ridge``  Ridge regression of the winsorised outcome (return units). L2, not
               L1: with tens of thousands of correlated n-grams, ridge is stable
               and predicts better; we do not need a sparse word list.
    ``logit``  L2 logistic regression of the sign of the reaction (up/down), for
               the classification view of task 7.
* Optional ``residualize=True``: train on the part of the outcome the SUE deciles
  do not explain (deciles fitted on the same training window). That model asks
  directly "what does the text add beyond the surprise?".
* Out of sample: chronological cross-fitting (crossfit.py). The vocabulary, the
  idf weights and the firm filter are fitted on the training window of each fold
  only, never on the evaluation years.
* Hyper-parameter (penalty) chosen by mean within-season IC of the *dev*
  pseudo-out-of-sample predictions, never on validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

import settings as S
from crossfit import chronological_folds
from metrics import rank_ic
from text_utils import TFIDF_TOKEN_PATTERN, mask_numbers


@dataclass
class TfidfConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 20
    max_df: float = 0.5
    max_features: int = 50_000
    min_firms: int = 20
    ridge_alphas: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)
    logit_Cs: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
    min_train_docs: int = 500
    residualize: bool = False
    extra: dict = field(default_factory=dict)


def _vectorize(train_docs, test_docs, train_firms, cfg: TfidfConfig):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(token_pattern=TFIDF_TOKEN_PATTERN, ngram_range=cfg.ngram_range,
                          min_df=min(cfg.min_df, max(1, len(train_docs) // 50)), max_df=cfg.max_df,
                          max_features=cfg.max_features, sublinear_tf=True, dtype=np.float32,
                          lowercase=False)          # mask_numbers already lower-cases
    Xtr = vec.fit_transform(train_docs)
    vocab = np.array(vec.get_feature_names_out())

    # Firm filter: number of distinct firms using each term in the training window.
    firms = pd.factorize(np.asarray(train_firms))[0]
    G = sp.csr_matrix((np.ones(len(firms)), (firms, np.arange(len(firms)))),
                      shape=(firms.max() + 1, len(firms)))
    n_firms = np.asarray(((G @ (Xtr > 0).astype(np.float32)) > 0).sum(axis=0)).ravel()
    keep = n_firms >= min(cfg.min_firms, max(2, firms.max() // 10))
    Xte = vec.transform(test_docs)
    return Xtr[:, keep], Xte[:, keep], vocab[keep]


def _decile_means(train: pd.DataFrame, y: str) -> pd.Series:
    return train.groupby("sue_decile")[y].mean()


def tfidf_signals(panel: pd.DataFrame, texts: pd.DataFrame, outcome: str = S.PRIMARY_OUTCOME,
                  cfg: TfidfConfig | None = None, eval_splits=("dev", "validation"),
                  return_details: bool = False):
    """
    Cross-fitted TF-IDF ridge and logit signals for one outcome.

    panel: analysis panel (panel.build_panel); texts: accession, text_body.
    Returns a frame indexed like ``panel`` with ``tfidf_ridge__<outcome>`` and
    ``tfidf_logit__up``, and (if return_details) the tuning table and the top
    terms of the last fold's model.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    cfg = cfg or TfidfConfig()
    target = outcome + "_w" if outcome + "_w" in panel.columns else outcome
    docs = dict(zip(texts["accession"], texts["text_body"].map(mask_numbers)))
    has = panel["accession"].isin(docs.keys()) & panel[target].notna()
    pn = panel.loc[has]
    print(f"TF-IDF: {len(pn):,} events with text and outcome")

    ridge_name, logit_name = f"tfidf_ridge__{outcome}", "tfidf_logit__up"
    preds_r = {a: pd.Series(np.nan, index=panel.index) for a in cfg.ridge_alphas}
    preds_l = {c: pd.Series(np.nan, index=panel.index) for c in cfg.logit_Cs}
    fold_of = pd.Series(pd.NA, index=panel.index, dtype=object)
    last_models: dict = {}

    for es in eval_splits:
        for f in chronological_folds(pn, es):
            if len(f.train_idx) < cfg.min_train_docs:
                print(f"  {f.label}: {len(f.train_idx)} training documents < {cfg.min_train_docs}, skipped")
                continue
            tr, te = pn.loc[f.train_idx], pn.loc[f.test_idx]
            Xtr, Xte, vocab = _vectorize([docs[a] for a in tr["accession"]],
                                         [docs[a] for a in te["accession"]], tr["permno"], cfg)
            y = tr[target].to_numpy(float)
            offset_te = 0.0
            if cfg.residualize:
                dm = _decile_means(tr, target)
                y = y - tr["sue_decile"].map(dm).fillna(0).to_numpy()
                offset_te = 0.0   # the signal is the text component only
            up = (tr[S.PRIMARY_OUTCOME] > 0).astype(int).to_numpy()
            for a in cfg.ridge_alphas:
                m = Ridge(alpha=a).fit(Xtr, y)
                preds_r[a].loc[te.index] = m.predict(Xte) + offset_te
                if es == eval_splits[-1]:
                    last_models[a] = (m, vocab)     # kept to list the top terms
            for c in cfg.logit_Cs:
                lg = LogisticRegression(C=c, max_iter=2000).fit(Xtr, up)
                preds_l[c].loc[te.index] = lg.decision_function(Xte)
            fold_of.loc[te.index] = f.label
            print(f"  {f.label}: train {len(tr):,} docs, {Xtr.shape[1]:,} terms, test {len(te):,}")

    if fold_of.notna().sum() == 0:
        print("TF-IDF: no fold had enough training documents; signals left empty "
              "(expected until part 5 has cleaned several years of releases)")
        out = pd.DataFrame({ridge_name: np.nan, logit_name: np.nan, "tfidf_fold": fold_of},
                           index=panel.index)
        return (out, {"tuning": pd.DataFrame(), "top_terms": pd.DataFrame(), "alpha": None,
                      "C": None}) if return_details else out

    # Tune on dev pseudo-OOS predictions only (never validation / holdout).
    is_dev = panel["split"].eq("dev") & fold_of.notna()
    tune = []
    for a, p in preds_r.items():
        d = panel.loc[is_dev].assign(_s=p[is_dev])
        tune.append({"model": "ridge", "param": a, **rank_ic(d, "_s", outcome)})
    for c, p in preds_l.items():
        d = panel.loc[is_dev].assign(_s=p[is_dev])
        tune.append({"model": "logit", "param": c, **rank_ic(d, "_s", S.PRIMARY_OUTCOME)})
    tune = pd.DataFrame(tune)

    def best(model, default):
        t = tune[(tune["model"] == model) & tune["mean_ic"].notna()]
        return t.sort_values("mean_ic").iloc[-1]["param"] if len(t) else default

    a_best = best("ridge", cfg.ridge_alphas[len(cfg.ridge_alphas) // 2])
    c_best = best("logit", cfg.logit_Cs[len(cfg.logit_Cs) // 2])
    print(f"TF-IDF: chosen on dev pseudo-OOS IC: ridge alpha={a_best}, logit C={c_best}")

    out = pd.DataFrame(index=panel.index)
    out[ridge_name] = preds_r[a_best]
    out[logit_name] = preds_l[c_best]
    out["tfidf_fold"] = fold_of
    if not return_details:
        return out

    top = pd.DataFrame()
    if a_best in last_models:
        a = a_best
        m, vocab = last_models[a]
        coef = pd.Series(m.coef_, index=vocab).sort_values()
        top = pd.concat([coef.head(30).rename("coef").to_frame().assign(direction="negative"),
                         coef.tail(30).iloc[::-1].rename("coef").to_frame().assign(direction="positive")])
        top.index.name = "term"
        top = top.reset_index()
        top["alpha"] = a
    return out, {"tuning": tune, "top_terms": top, "alpha": a_best, "C": c_best}
