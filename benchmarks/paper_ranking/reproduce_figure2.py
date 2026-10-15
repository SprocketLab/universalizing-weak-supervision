"""Reproduce the paper's real-data ranking experiment (Figure 2: end-model Kendall tau vs number of LFs).

Setup from the paper (Sec. 5 and Appendix H): d = 5 items per set, 5000 training / 1000 test sets, simulated
Mallows LFs with 1/3 good (theta ~ U(0.2, 1)) and 2/3 bad (theta ~ U(0.001, 0.01)), PT-Ranking ListMLE end
model, test Kendall tau distance (lower is better). Item sets are drawn exactly as the original code did
(``legacy_sampling``).

Methods:
  legacy_ours   paper code: triplet_opt + weighted "Kemeny" (medoid), legacy=True
  legacy_mv     paper code: unweighted "Kemeny" (medoid) = the paper's majority-vote baseline
  snorkel       Snorkel LabelModel with every permutation as a class (the paper's Snorkel baseline)
  new_ours      uws 1.0: triplet_opt + true weighted Kemeny
  new_sign      uws 1.0: sign_triplet_opt + true weighted Kemeny
  sup_20/50/100 end model trained on that fraction of the true training labels

Example (one job = one dataset, one seed, one number of LFs; fully supervised runs use --m 0)::

    python reproduce_figure2.py --dataset boardgames --seed 0 --m 6 --out results/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from uws.datasets import create_dataset
from uws.experiments.ptranking_wrapper import PtrankingWrapper
from uws.experiments.real_workflow import get_weak_labels
from uws.ranking import RankingUtils

REPO = Path(__file__).resolve().parents[2]
METHODS = {
    "legacy_ours": dict(legacy=True, train_method="triplet_opt", inference_rule="weighted_kemeny"),
    "legacy_mv": dict(legacy=True, train_method="triplet_opt", inference_rule="kemeny"),
    "snorkel": dict(legacy=True, train_method="triplet_opt", inference_rule="snorkel"),
    "new_ours": dict(legacy=False, train_method="triplet_opt", inference_rule="weighted_kemeny"),
    "new_sign": dict(legacy=False, train_method="sign_triplet_opt", inference_rule="weighted_kemeny"),
}
PAPER_LFS = dict(good_min_theta=0.2, good_max_theta=1.0, bad_min_theta=0.001, bad_max_theta=0.01)


def train_end_model(conf, dataset, Y_train_override, work, tag, frac=1.0):
    """Train the ListMLE end model; return final-epoch test metrics."""
    dc, w, l2r = conf["data_conf"], conf["weak_sup_conf"], copy.deepcopy(conf["l2r_training_conf"])
    if Y_train_override is not None:
        dataset.set_Y_tilde(Y_train_override)
        l2r["use_weak_labels"] = True
    else:
        l2r["use_weak_labels"] = False
    X_train, X_test, Y_train, Y_test = dataset.get_train_test_torch(l2r["use_weak_labels"])
    k = int(round(frac * len(X_train)))
    pw = PtrankingWrapper({**dc, "project_root": str(work)}, w, l2r, f"res/{tag}")
    pw.set_data(X_train[:k], X_test, Y_train[:k], Y_test)
    res = pw.train_model(pw.get_model())
    last = lambda key: float(torch.as_tensor(res[key][-1]).reshape(-1)[0])  # noqa: E731
    return {"test_tau": last("test_tau"), "test_ndcg1": last("test_ndcg1"), "test_ndcg5": last("test_ndcg5")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["boardgames", "movies"], required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed of the LF thetas and LF samples")
    ap.add_argument("--m", type=int, required=True, help="number of LFs (0 = fully supervised baselines)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--work", default="work", help="scratch directory for caches")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    conf = yaml.safe_load(open(REPO / "configs" / f"{a.dataset}_ranking_experiment.yaml"))
    work = Path(a.work).resolve()
    (work / "data").mkdir(parents=True, exist_ok=True)
    data_link = work / "data" / Path(conf["data_conf"]["raw_data_path"]).name
    if not data_link.exists():
        os.symlink(REPO / conf["data_conf"]["raw_data_path"], data_link)
    dc = conf["data_conf"]
    dc.update(project_root=str(work), raw_data_path=f"data/{data_link.name}", n_train=a.n_train,
              n_test=a.n_test, processed_data_path=f"proc/{a.dataset}_{a.n_train}_{a.n_test}",
              recreate_if_exists=False, legacy_sampling=True)
    conf["project_root"] = str(work)
    dataset = create_dataset(dc)
    dataset.create_samples()
    r_utils = RankingUtils(dataset.d)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.m == 0:
        for frac in (0.2, 0.5, 1.0):
            t0 = time.time()
            r = train_end_model(conf, dataset, None, work, f"{a.dataset}_sup{frac}_s{a.seed}", frac)
            rows.append({"dataset": a.dataset, "seed": a.seed, "m": 0, "method": f"sup_{int(frac * 100)}",
                         "weak_kt": None, **r, "seconds": time.time() - t0})
            print(rows[-1], flush=True)
    else:
        # generate the shared LFs exactly as the paper code did (legacy theta draws and sampler)
        get_weak_labels(dataset, {**conf["weak_sup_conf"], **METHODS["legacy_ours"], **PAPER_LFS, "num_LFs": a.m,
                                  "seed": a.seed, "synthetic": True, "recreate_if_exists": True,
                                  "checkpoint_path": f"lfs/{a.dataset}_m{a.m}_s{a.seed}"})
        for name in a.methods:
            t0 = time.time()
            w = {**conf["weak_sup_conf"], **METHODS[name], **PAPER_LFS, "num_LFs": a.m, "seed": a.seed,
                 "synthetic": True, "recreate_if_exists": False,
                 # every method aggregates the same LF outputs (generated once, with the paper's sampler)
                 "checkpoint_path": f"lfs/{a.dataset}_m{a.m}_s{a.seed}"}
            conf_m = {**conf, "weak_sup_conf": w}
            Y_hat, thetas, ikt = get_weak_labels(dataset, w)
            weak_kt = r_utils.mean_kt_distance(Y_hat[: dataset.n_train], dataset.Y_train)
            r = train_end_model(conf_m, dataset, Y_hat, work, f"{a.dataset}_{name}_m{a.m}_s{a.seed}")
            rows.append({"dataset": a.dataset, "seed": a.seed, "m": a.m, "method": name, "weak_kt": weak_kt,
                         "lf_kt": ikt[0], **r, "seconds": time.time() - t0})
            print({k: v for k, v in rows[-1].items() if k != "lf_kt"}, flush=True)
    with open(out / f"{a.dataset}_m{a.m}_s{a.seed}.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
