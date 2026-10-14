# Universalizing Weak Supervision

[![CI](https://github.com/Sala-Group/universalizing-weak-supervision/actions/workflows/ci.yml/badge.svg)](https://github.com/Sala-Group/universalizing-weak-supervision/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/universalizing-weak-supervision/badge/?version=latest)](https://universalizing-weak-supervision.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2112.03865-b31b1b.svg)](https://arxiv.org/abs/2112.03865)

![framework](assets/uws_figure.png)

`uws` is the library behind the ICLR 2022 paper
[Universalizing Weak Supervision](https://arxiv.org/abs/2112.03865) by Changho Shin, Winfred Li,
Harit Vishwakarma, Nicholas Roberts, and Frederic Sala. It gives weak supervision for **any label type**:
labeling functions (LFs) are modeled as exponential families around the latent label in a metric space,
their parameters are estimated without labels by a triplet method, and labels are inferred by weighted
Fréchet means.

| Label type | Label model | Inference |
|---|---|---|
| Full / partial rankings | `uws.ranking.RankingLabelModel` (Mallows LFs) | weighted Kemeny, pairwise majority, position estimation |
| Real values | `uws.regression.ContinuousLabelModel` (Gaussian LFs) | closed-form posterior mean |
| Finite metric spaces | `uws.metric.MetricLabelModel` | weighted Fréchet mean |
| Manifolds (e.g. hyperbolic) | `uws.geodesic.GeodesicLabelModel` | Riemannian Fréchet mean |
| Binary | `uws.binary.BinaryLabelModel` (FlyingSquid) | |

## Installation

```bash
pip install uws                 # core: rankings, regression, metric spaces
pip install "uws[geodesic]"     # + manifold-valued labels (torch, geoopt)
pip install "uws[l2r]"          # + learning-to-rank end models (torch; vendored PT-Ranking)
pip install "uws[all]"          # everything, including the Snorkel baseline
```

From source: `git clone` this repository, then `pip install -e ".[dev,all]"`.
Requires Python ≥ 3.9.

## Quickstart

```python
import numpy as np
from uws.ranking import RankingLabelModel, RankingUtils, generate_true_rankings, sample_mallows_LFs

rng = np.random.default_rng(0)
r_utils = RankingUtils(d=5)                                  # rankings over 5 items
Y = generate_true_rankings(5, 1000, rng, r_utils)            # latent rankings (unknown in practice)
L = sample_mallows_LFs(Y, 6, [0.2, 0.3, 2.0, 2.5, 0.4, 3.0], rng=rng)  # 6 noisy ranking LFs

lm = RankingLabelModel(r_utils, train_method="median_triplet_opt", inference_rule="weighted_kemeny")
lm.fit(L)                                                    # no labels used
Y_hat = lm.predict(L)
print(lm.thetas)                                             # estimated LF accuracies
print(r_utils.mean_kt_distance(Y_hat, Y))                    # vs. unweighted:
print(r_utils.mean_kt_distance(lm.predict(L, inference_rule="kemeny"), Y))
```

Real-valued labels:

```python
from uws.regression import ContinuousLabelModel
lm = ContinuousLabelModel().fit(L_train, var_Y)   # L_train: (n, m) LF outputs; var_Y: label variance
y_hat, var = lm.predict(L_test, return_var=True)
```

Any finite metric space (here, a graph with shortest-path distances `D`, and `-1` for abstains):

```python
from uws.metric import MetricLabelModel
lm = MetricLabelModel(D).fit(L)        # L: (n, m) integer LF outputs
y_hat = lm.predict(L)
proba = lm.predict_proba(L)
```

All label models share the same interface: `fit(L)`, `predict(L)`, `get_weights()`, `save()` / `load()`.

## Reproducing the paper

The tutorial notebooks in [`tutorials/`](tutorials) reproduce the experiments:

* Full and partial rankings on real data: `tutorials/{boardgames,movies}/RankingExperiments.ipynb`,
  with configs in `configs/{boardgames,movies}_ranking_experiment.yaml`. The main knobs are `n_train`, `n_test`,
  `p` (observation probability for partial rankings: `null | 0.2 | 0.4 | 0.6 | 0.8`), `num_LFs` and
  `inference_rule` (`weighted_kemeny`, `kemeny`, `pairwise_majority`, `weighted_pairwise_majority`, or
  `snorkel` for the baseline).
* Synthetic rankings: `tutorials/synthetic/{Full,Partial}-Rankings-Experiments-Center-Recovery.ipynb`
  and a minimal walk-through in `tutorials/synthetic/Ranking-Example.ipynb`.
* Regression: `tutorials/{boardgames,movies}/RegressionExperiments.ipynb`, `tutorials/synthetic/Regression-Experiments.ipynb`.
* Geodesic regression: `tutorials/geodesic-regression/geodesic_regression.ipynb`.
* Generic metric spaces: `tutorials/metric-spaces/generic_metric_spaces.ipynb`.

The real-data ranking pipeline can also run from the command line:

```bash
uws-experiment configs/boardgames_ranking_experiment.yaml --project-root . \
    --set weak_sup_conf.num_LFs=6 --set weak_sup_conf.p=0.4 [--train]
```

### Version 1.0 vs. the paper code

Version 1.0 fixes several bugs and corrects some approximations of the original research code (see
[CHANGELOG.md](CHANGELOG.md)). Most visibly:

* `kemeny` now returns the true (weighted) Kemeny ranking. The paper's rule picked the best of the LF
  rankings and is available as `kemeny_medoid`.
* The partial-ranking Kendall distance is Fagin et al.'s $K^{(p)}$.
* Mallows sampling is exact (repeated insertion model).
* The Gaussian label model no longer inverts ill-conditioned matrices.

Pass `legacy=True` to `RankingUtils`, `ContinuousLabelModel` or `GeodesicLabelModel`, and
`sampler="legacy"` to `Mallows`, to reproduce the paper's numbers.

## Development

```bash
pip install -e ".[dev,all]"
pre-commit install
pytest
ruff check uws tests
```

Documentation: `pip install -e ".[docs]" && sphinx-build docs docs/_build`.

## Citation

```bibtex
@inproceedings{shin2022universalizing,
  title={Universalizing Weak Supervision},
  author={Shin, Changho and Li, Winfred and Vishwakarma, Harit and Roberts, Nicholas Carl and Sala, Frederic},
  booktitle={The Tenth International Conference on Learning Representations},
  year={2022}
}
```

## License

Apache-2.0. `uws/_vendor/ptranking` is a modified copy of [PT-Ranking](https://github.com/wildltr/ptranking)
(MIT License).
