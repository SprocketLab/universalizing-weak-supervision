"""How well do the label-free estimators recover each LF's accuracy? (Figure 2 sweep LFs.)

For every (dataset, seed, number of LFs) run of ``reproduce_figure2.py``, re-fit the ranking label model on the
saved LF outputs with each estimator and compare its estimates with the truth:

* the true Mallows theta of each LF (redrawn from the run's seed exactly as the sweep drew it), and
* each LF's empirical mean Kendall distance to the true rankings (``lf_kt`` in the results JSON).

Estimated thetas are inverted from the estimated distances over ``[1e-4, 50]`` so the plot shows what the
estimator believes; the label model itself floors theta at 0.1 before using it as a weight (dashed line).

Usage::

    python theta_recovery.py --lfs <dir with job_*/lfs/*/synthetic_weak_labels.pkl> --results results/ --out figures/
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uws.label_model import RankingLabelModel  # noqa: E402
from uws.ranking import Mallows, RankingUtils  # noqa: E402

PAPER_LFS = dict(good=(0.2, 1.0), bad=(0.001, 0.01))
ESTIMATORS = {
    "triplet_opt": "additive triplet (paper)",
    "median_triplet_opt": "additive median triplet",
    "sign_triplet_opt": "pair-sign rank-1 fit",
}
COLORS = {"triplet_opt": "tab:orange", "median_triplet_opt": "tab:green", "sign_triplet_opt": "tab:blue"}


def true_thetas(m: int, seed: int) -> np.ndarray:
    """The sweep's thetas: ``generate_synthetic_LFs(..., legacy_thetas=True)`` with the paper's ranges."""
    rs = np.random.RandomState(seed)
    m_good = int(m / 3)
    return np.sort(np.concatenate([rs.uniform(*PAPER_LFS["good"], m_good), rs.uniform(*PAPER_LFS["bad"], m - m_good)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfs", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for pkl in sorted(Path(a.lfs).glob("job_*/lfs/*/synthetic_weak_labels.pkl")):
        dataset, m, seed = pkl.parent.name.split("_")
        m, seed = int(m[1:]), int(seed[1:])
        res = json.load(open(Path(a.results) / f"{dataset}_m{m}_s{seed}.json"))
        lf_kt = np.asarray(res[0]["lf_kt"])
        L = pickle.load(open(pkl, "rb"))
        r_utils = RankingUtils(len(L[0][0]))
        P = r_utils.d * (r_utils.d - 1) / 2
        theta = true_thetas(m, seed)
        for est in ESTIMATORS:
            lm = RankingLabelModel(r_utils, train_method=est).fit(L)
            est_theta = [Mallows.estimate_theta(r_utils.d, e, bounds=(1e-4, 50.0)) for e in lm.expected_dists]
            for j in range(m):
                rows.append(dict(dataset=dataset, m=m, seed=seed, lf=j, estimator=est, true_theta=theta[j],
                                 est_theta=float(est_theta[j]), weight=float(lm.thetas[j]), true_dist=float(lf_kt[j]),
                                 est_dist=float(lm.expected_dists[j] / P), good=bool(theta[j] >= 0.2)))
        print(pkl.parent.name, flush=True)
    json.dump(rows, open(out / "theta_recovery.json", "w"), indent=0)

    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(2, len(ESTIMATORS), figsize=(4.2 * len(ESTIMATORS), 8), squeeze=False)
    for k, est in enumerate(ESTIMATORS):
        R = [r for r in rows if r["estimator"] == est]
        ax = axes[0, k]
        for good, mk in ((True, "o"), (False, "x")):
            S = [r for r in R if r["good"] == good]
            ax.scatter([r["true_theta"] for r in S], [r["est_theta"] for r in S], s=12, marker=mk, alpha=0.6,
                       color=COLORS[est], label="good LFs" if good else "near-random LFs")
        ax.plot([1e-4, 10], [1e-4, 10], "k-", lw=0.8)
        ax.axhline(0.1, color="gray", ls="--", lw=0.8)
        ax.set(xscale="log", yscale="log", xlim=(5e-4, 3), ylim=(1e-4, 30), xlabel="true θ", ylabel="estimated θ",
               title=ESTIMATORS[est])
        ax.legend(fontsize=8, loc="upper left")
        ax = axes[1, k]
        for ds, mk in zip(datasets, ("o", "s")):
            S = [r for r in R if r["dataset"] == ds]
            ax.scatter([r["true_dist"] for r in S], [r["est_dist"] for r in S], s=12, marker=mk, alpha=0.5,
                       color=COLORS[est], label=ds)
        ax.plot([0, 0.6], [0, 0.6], "k-", lw=0.8)
        ax.set(xlim=(0, 0.6), ylim=(-0.05, 0.75), xlabel="actual LF distance to truth (normalized KT)",
               ylabel="estimated LF distance")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("LF accuracy recovery on the Figure 2 sweep LFs (2 datasets × 3 seeds × 6 LF counts)")
    fig.tight_layout()
    fig.savefig(out / "theta_recovery.png", dpi=150)

    # summary: error in normalized distance and ratio of mean weight (good / near-random)
    lines = ["| estimator | MAE distance, good LFs | MAE distance, near-random LFs | median θ, near-random LFs "
             "| mean weight good : near-random |", "|---|---:|---:|---:|---:|"]
    for est in ESTIMATORS:
        R = [r for r in rows if r["estimator"] == est]
        g = [r for r in R if r["good"]]
        b = [r for r in R if not r["good"]]
        mae = lambda S: np.mean([abs(r["est_dist"] - r["true_dist"]) for r in S])  # noqa: E731
        w = lambda S: np.mean([r["weight"] for r in S])  # noqa: E731
        lines.append(f"| `{est}` | {mae(g):.3f} | {mae(b):.3f} | {np.median([r['est_theta'] for r in b]):.3g} "
                     f"| {w(g) / w(b):.1f} |")
    (out / "theta_recovery.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
