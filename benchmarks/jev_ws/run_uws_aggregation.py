"""UWS label models as aggregators for the jev-ws fast-decision-model pool (exploratory).

Runs inside the jev-ws project (package ``csone``, cached LF predictions, jev-benchmarking metrics) and reuses
its cross-fitted protocol (``scripts/run_tuned_aggregation.py``: 2 folds by uid hash, label-dependent choices made
on one fold and applied to the other, thresholds tuned the same way). Label models are always fit on unlabeled
LF outputs only.

Aggregators (per head; a head kind a method does not cover falls back to ``uws_logpool``):

* ``uws_vote``     weighted vote with log-odds weights from label-free accuracy estimates. Under symmetric
                   K-class noise the centered agreement ``(K P(a=b) - 1)/(K-1) = s_a s_b``, which is fit as a rank-1
                   factorization over all LF pairs (``uws.utils.rank_one_fit``).
* ``uws_logpool``  weighted geometric pool of the LF probability vectors (same weights).
* ``uws_linpool``  weighted linear pool of the LF probability vectors (same weights).
* ``uws_gauss``    score heads: Gaussian label model (``uws.regression.ContinuousLabelModel``) on each LF's expected
                   score; binary heads: the same model on the LF log-odds.
* ``uws_ordinal``  score heads: ``uws.metric.MetricLabelModel`` with the ordinal metric ``|i - j|``.
* ``*_fam``        the same, with accuracies estimated only from cross-family LF pairs (LFs sharing a base model
                   make correlated errors, which violates the conditional-independence assumption).
* ``uws_tuned``    the UWS method selected on the other fold (like ``snorkel_tuned``).

Usage (on neuronic, inside the jev-ws environment)::

    source /n/fs/cogai/cs1095/jev-ws/env.sh
    PYTHONPATH=/path/to/uws:$PYTHONPATH CSONE_POOL=strong python run_uws_aggregation.py --tasks all
    python run_uws_aggregation.py --summary          # tables for both pools

Writes ``results/uws{suffix}/<task>.parquet`` (only new files) and ``results/uws/summary.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CSONE_ROOT", "/n/fs/cogai/cs1095/jev-ws"))
sys.path.insert(0, str(ROOT / "scripts"))

from csone import aggregate as agg  # noqa: E402
from csone.config import ALL_TASKS, CATEGORY, POOLS, RESULTS_DIR, TIER_OF, TIERS, lf_models, results_suffix  # noqa: E402
from csone.tasks import task as get_task  # noqa: E402
from run_tuned_aggregation import Tuner, shift  # noqa: E402

from uws.metric import MetricLabelModel  # noqa: E402
from uws.regression import ContinuousLabelModel  # noqa: E402
from uws.utils import rank_one_fit  # noqa: E402

FAMILY = {"laya-en": "laya", "laya-multi": "laya", "von": "von", "kev-0.8b": "qwen", "kev-4b": "qwen",
          "kev-9b": "qwen", "nimble-9b": "qwen", "nimble-9b-v3": "qwen", "jev-gemma4-26b": "gemma",
          "rune-26b": "gemma", "decider-gemma4-31b": "gemma"}
EPS = 1e-6


# ------------------------------------------------------------------------------------------------ label-free weights
def vote_accuracies(hd, groups=None) -> np.ndarray:
    """Per-LF accuracy from hard votes via the rank-1 centered-agreement model."""
    L = hd.L
    N, M = L.shape
    K = max(2.0, float(hd.allowed.sum(1).mean()))
    agree = np.zeros((M, M))
    cnt = np.zeros((M, M))
    for a in range(M):
        for b in range(M):
            ok = (L[:, a] >= 0) & (L[:, b] >= 0)
            cnt[a, b] = ok.sum()
            agree[a, b] = (L[ok, a] == L[ok, b]).mean() if ok.any() else 0.0
    A = (K * agree - 1) / (K - 1)
    W = cnt.copy()
    if groups is not None:
        g = np.asarray(groups)
        W[g[:, None] == g[None, :]] = 0.0
    s = rank_one_fit(A, W)
    acc = (1 + (K - 1) * s) / K
    return np.clip(acc, 1 / K + 1e-3, 1 - 1e-3), K


def log_odds_weights(acc: np.ndarray, K: float) -> np.ndarray:
    return np.log(acc * (K - 1) / (1 - acc))


# ------------------------------------------------------------------------------------------------ aggregators
def uws_vote(hd, groups=None):
    acc, K = vote_accuracies(hd, groups)
    w = log_odds_weights(acc, K)
    N, Kc = hd.allowed.shape
    S = np.zeros((N, Kc))
    for j in range(hd.L.shape[1]):
        ok = hd.L[:, j] >= 0
        np.add.at(S, (np.nonzero(ok)[0], hd.L[ok, j]), w[j])
    S = S + 1e-6 * hd.P.mean(1)  # tie-break like csone's majority
    S = np.where(hd.allowed, S, -np.inf)
    S = S - S.max(1, keepdims=True)
    return agg._mask(np.exp(S), hd.allowed)


def _pool_weights(hd, groups=None):
    acc, K = vote_accuracies(hd, groups)
    w = np.maximum(log_odds_weights(acc, K), 0.0) + 1e-3
    return w / w.sum()


def uws_logpool(hd, groups=None):
    w = _pool_weights(hd, groups)
    logp = (w[None, :, None] * np.log(hd.P + EPS)).sum(1)
    logp = np.where(hd.allowed, logp, -np.inf)
    logp = logp - logp.max(1, keepdims=True)
    return agg._mask(np.exp(logp), hd.allowed)


def uws_linpool(hd, groups=None):
    w = _pool_weights(hd, groups)
    return agg._mask((w[None, :, None] * hd.P).sum(1), hd.allowed)


def _gaussian_levels(mu, var, hd):
    K = hd.allowed.shape[1]
    k = np.arange(K)[None, :]
    sd2 = np.maximum(var, 1e-2)[:, None] if np.ndim(var) else max(float(var), 1e-2)
    logits = -((k - mu[:, None]) ** 2) / (2 * sd2)
    logits = np.where(hd.allowed, logits, -np.inf)
    logits -= logits.max(1, keepdims=True)
    return agg._mask(np.exp(logits), hd.allowed)


def _fit_gauss(X, groups=None):
    """Gaussian LM on continuous LF outputs X (N x M); var_Y set label-free to the mean LF variance (it only
    scales the centered predictions)."""
    var_Y = float(np.mean(X.var(0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm = ContinuousLabelModel().fit(X, var_Y)
        return lm.predict(X, return_var=True)


def uws_gauss(hd, groups=None):
    if hd.kind == "score":
        levels = np.arange(hd.allowed.shape[1])
        P = np.where(hd.allowed[:, None, :], hd.P, 0.0)
        X = (P * levels).sum(-1) / np.maximum(P.sum(-1), EPS)  # expected score per LF
        mu, var = _fit_gauss(X)
        return _gaussian_levels(mu, var, hd)
    if hd.kind == "binary":
        p = np.clip(hd.P[:, :, 1], 1e-4, 1 - 1e-4)
        mu, _ = _fit_gauss(np.log(p / (1 - p)))
        q = 1 / (1 + np.exp(-mu))
        return np.stack([1 - q, q], 1)
    return uws_logpool(hd, groups)


def uws_ordinal(hd, groups=None, center_power: float = 1.0):
    if hd.kind != "score":
        return uws_logpool(hd, groups)
    K = hd.allowed.shape[1]
    D = np.abs(np.subtract.outer(np.arange(K), np.arange(K))).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm = MetricLabelModel(D, center_power=center_power).fit(hd.L, groups=groups)
    return agg._mask(lm.predict_proba(hd.L), hd.allowed)


BASE_METHODS = {"uws_vote": uws_vote, "uws_logpool": uws_logpool, "uws_linpool": uws_linpool,
                "uws_gauss": uws_gauss, "uws_ordinal": uws_ordinal}


# ------------------------------------------------------------------------------------------------ protocol
def method_dists(tu: Tuner, fn, groups) -> dict:
    out = {}
    for h, hd in tu.td.heads.items():
        if hd.allowed.shape[1] < 2:
            out[h] = agg._mask(np.ones(hd.allowed.shape), hd.allowed)
            continue
        try:
            out[h] = fn(hd, groups)
        except (ValueError, np.linalg.LinAlgError) as e:  # e.g. no usable cross-family partner
            logging.info("%s/%s: %s failed (%s); using prob-avg", tu.name, h, fn.__name__, str(e)[:80])
            out[h] = agg.prob_average(hd)
    return out


def run_task(name: str) -> list[dict]:
    sources, _ = agg.pick_sources(name, lf_models())
    if len(sources) < 3:
        return []
    tu = Tuner(name, sources)
    t = get_task(name)
    fams = [FAMILY.get(m, m) for m, _ in sources]
    use_fam = len(set(fams)) > 1 and all(sum(f != g for g in fams) >= 1 for f in fams)

    def row(method, res, **extra):
        return {"task": name, "tier": TIER_OF[name], "category": CATEGORY[name], "method": method,
                "method_type": "uws", "n_examples": len(tu.td.exs), "primary_metric": f"{t.primary.head}:{t.primary.metric}",
                "primary": res["primary"], "metrics": json.dumps(res.get("metrics", {}), default=float), **extra}

    raw = {}
    for mname, fn in BASE_METHODS.items():
        raw[mname] = method_dists(tu, fn, None)
        if use_fam:
            raw[mname + "_fam"] = method_dists(tu, fn, fams)
    rows = []
    thr = {m: tu.thresholded(d) for m, d in raw.items()}
    for m, d in thr.items():
        rows.append(row(m, tu.score_all(d)))

    # uws_tuned: pick the method on fold A (thresholds tuned within A), apply to fold B
    final, picked = {}, {}
    for f in (0, 1):
        tr = np.flatnonzero(tu.folds != f)
        scores = {}
        for m, d in raw.items():
            dd = {}
            for h, x in d.items():
                if tu.td.heads[h].kind == "binary":
                    th = tu.thr_fn(tu.gold[h][tr], x[tr, 1])
                    p = shift(x[:, 1], th)
                    dd[h] = np.stack([1 - p, p], axis=1)
                else:
                    dd[h] = x
            scores[m] = (tu.primary(dd, tr), dd)
        best = max(scores, key=lambda m: np.nan_to_num(scores[m][0], nan=-1))
        picked[f] = best
        for h, x in scores[best][1].items():
            final.setdefault(h, np.zeros_like(x))
            final[h][tu.folds == f] = x[tu.folds == f]
    rows.append(row("uws_tuned", tu.score_all(final), chosen=json.dumps(picked)))
    return rows


# ------------------------------------------------------------------------------------------------ summary
BASELINES = ["best_single_cv", "majority", "prob_avg", "snorkel_tuned"]


def md_table(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> markdown (avoids the optional tabulate dependency)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:.2f}" if isinstance(v, float) else str(v) for v in r]) + " |")
    return "\n".join(lines)


def summarize() -> str:
    out = ["# UWS label models on the jev-ws fast-decision pools", "",
           "Primary metric ×100, mean over tasks, cross-fitted (2-fold) protocol of `run_tuned_aggregation.py`. "
           "Wins = tasks where the method beats the column method.", ""]
    for pool in POOLS:
        sfx = results_suffix(pool)
        base_dir, uws_dir = RESULTS_DIR / f"tuned{sfx}", RESULTS_DIR / f"uws{sfx}"
        if not uws_dir.exists():
            continue
        base = pd.concat([pd.read_parquet(p) for p in sorted(base_dir.glob("*.parquet"))])
        uws = pd.concat([pd.read_parquet(p) for p in sorted(uws_dir.glob("*.parquet"))])
        best_fixed = (base[base.method_type == "individual_tuned"].groupby("method").primary.mean().idxmax())
        allr = pd.concat([base, uws])
        piv = allr.pivot_table(index="task", columns="method", values="primary", aggfunc="first")
        piv = piv.dropna(subset=["uws_tuned"])
        cols = BASELINES + [best_fixed] + sorted(c for c in uws.method.unique())
        out += [f"## Pool `{pool}` ({len(piv)} tasks; best fixed model: `{best_fixed}`)", "",
                "| method | mean | vs prob_avg | wins vs prob_avg | vs snorkel_tuned | wins vs snorkel_tuned | "
                "vs best_single_cv | vs " + best_fixed + " |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in cols:
            if c not in piv:
                continue
            d = lambda ref: 100 * (piv[c] - piv[ref]).mean()  # noqa: E731
            w = lambda ref: int((piv[c] > piv[ref] + 1e-9).sum())  # noqa: E731
            out.append(f"| {c} | {100 * piv[c].mean():.2f} | {d('prob_avg'):+.2f} | {w('prob_avg')} | "
                       f"{d('snorkel_tuned'):+.2f} | {w('snorkel_tuned')} | {d('best_single_cv'):+.2f} | "
                       f"{d(best_fixed):+.2f} |")
        cat = allr[allr.task.isin(piv.index)].assign(primary=lambda x: 100 * x.primary)
        cat = cat[cat.method.isin(["prob_avg", "snorkel_tuned", "uws_tuned", "uws_gauss", "uws_logpool"])]
        tab = cat.pivot_table(index="category", columns="method", values="primary", aggfunc="mean").round(2)
        out += ["", "By category (mean primary ×100):", "", md_table(tab), ""]
        judge = piv.loc[[t for t in piv.index if CATEGORY[t] in ("Judge", "Safety", "Legal")]]
        keep = [c for c in ["best_single_cv", "prob_avg", "snorkel_tuned", "uws_tuned", "uws_gauss", "uws_gauss_fam",
                            "uws_ordinal", "uws_logpool"] if c in judge]
        out += ["Per task, Judge/Safety/Legal (where Snorkel struggled):", "",
                md_table((100 * judge[keep]).round(2)), ""]
        picks = uws[uws.method == "uws_tuned"].chosen.map(json.loads)
        counts = pd.Series([v for p in picks for v in p.values()]).value_counts()
        out += ["`uws_tuned` selections over folds: " + ", ".join(f"{k} {v}" for k, v in counts.items()), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--index", type=int, default=None, help="run only the i-th task (Slurm array)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if a.summary:
        text = summarize()
        (RESULTS_DIR / "uws").mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "uws" / "summary.md").write_text(text)
        print(text)
        return
    names = [t for n in a.tasks for t in (TIERS.get(n) or (ALL_TASKS if n == "all" else [n]))]
    if a.index is not None:
        names = names[a.index : a.index + 1]
    out = RESULTS_DIR / f"uws{results_suffix()}"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.monotonic()
        try:
            rows = run_task(name)
        except Exception:  # noqa: BLE001
            logging.exception("%s failed", name)
            continue
        if rows:
            pd.DataFrame(rows).to_parquet(out / f"{name}.parquet")
            r = {x["method"]: x["primary"] for x in rows}
            logging.info("%s: uws_tuned=%.4f uws_logpool=%.4f uws_gauss=%.4f (%.0fs)", name, r["uws_tuned"],
                         r["uws_logpool"], r["uws_gauss"], time.monotonic() - t0)


if __name__ == "__main__":
    main()
