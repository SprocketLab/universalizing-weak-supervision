"""``uws-experiment``: run the real-data ranking pipeline from a YAML config.

Example::

    uws-experiment configs/boardgames_ranking_experiment.yaml --project-root . --set weak_sup_conf.p=0.4
"""

from __future__ import annotations

import argparse
import json
import os
import time

import yaml


def _set(conf: dict, dotted: str, value: str):
    keys = dotted.split(".")
    for k in keys[:-1]:
        conf = conf.setdefault(k, {})
    conf[keys[-1]] = yaml.safe_load(value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML with data_conf, weak_sup_conf and l2r_training_conf")
    ap.add_argument("--project-root", default=None, help="Root that relative paths in the config refer to")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config entry, e.g. weak_sup_conf.num_LFs=6")
    ap.add_argument("--train", action="store_true", help="Also train the learning-to-rank end model")
    ap.add_argument("--output", default=None, help="Where to write the JSON summary")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        _set(conf, k, v)
    root = args.project_root or conf.get("project_root") or "."
    conf["project_root"] = conf["data_conf"]["project_root"] = root

    from ..datasets import create_dataset
    from ..ranking.ranking import RankingUtils
    from .real_workflow import get_weak_labels

    t0 = time.time()
    dataset = create_dataset(conf["data_conf"])
    dataset.create_samples()
    Y_hat, thetas, individual_kt = get_weak_labels(dataset, conf["weak_sup_conf"])
    r_utils = RankingUtils(dataset.d)
    summary = {
        "config": conf,
        "weak_label_kt_train": r_utils.mean_kt_distance(Y_hat[: dataset.n_train], dataset.Y_train),
        "weak_label_kt_all": r_utils.mean_kt_distance(Y_hat, dataset.Y),
        "individual_lf_kt": individual_kt[0],
        "thetas": [float(t) for t in thetas],
    }
    if args.train:
        from .ptranking_wrapper import PtrankingWrapper

        dataset.set_Y_tilde(Y_hat)
        l2r = conf["l2r_training_conf"]
        X_train, X_test, Y_train, Y_test = dataset.get_train_test_torch(l2r["use_weak_labels"])
        wrapper = PtrankingWrapper(conf["data_conf"], conf["weak_sup_conf"], l2r, conf["results_path"],
                                   summary["weak_label_kt_all"], individual_kt)
        wrapper.set_data(X_train, X_test, Y_train, Y_test)
        result = wrapper.train_model(wrapper.get_model())
        summary["l2r"] = {k: v for k, v in result.items() if isinstance(v, (int, float, str, list))}
    summary["seconds"] = time.time() - t0

    out = args.output or os.path.join(root, conf.get("results_path", "."), "summary.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ("weak_label_kt_train", "weak_label_kt_all", "thetas")}, indent=2))
    print("Summary written to", out)


if __name__ == "__main__":
    main()
