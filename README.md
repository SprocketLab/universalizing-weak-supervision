# Universalizing Weak Supervision

[![CI](https://github.com/SprocketLab/universalizing-weak-supervision/actions/workflows/ci.yml/badge.svg)](https://github.com/SprocketLab/universalizing-weak-supervision/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/universalizing-weak-supervision/badge/?version=latest)](https://universalizing-weak-supervision.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2112.03865-b31b1b.svg)](https://arxiv.org/abs/2112.03865)

![framework](assets/uws_figure.png)

`uws` is a weak supervision library for labels that are not just classes: rankings, real numbers, nodes
of a graph, points in hyperbolic space, and more. It accompanies the ICLR 2022 paper
[Universalizing Weak Supervision](https://arxiv.org/abs/2112.03865).

## Contents

- [Motivation](#motivation)
- [Installation](#installation)
- [Sample Usage](#sample-usage)
- [Tutorials](#tutorials)
- [Reproducing the Paper](#reproducing-the-paper)
- [References](#references)
- [Release Notes](#release-notes)
- [Developer Guidelines](#developer-guidelines)
- [Citation](#citation)

## Motivation

Weak supervision replaces hand-labeled training data with labeling functions: heuristics, existing
models, or crowd workers that each produce noisy labels. A label model learns how accurate each labeling
function is, without seeing any ground truth, and combines their votes into training labels. Systems such
as Snorkel and Snorkel MeTaL made this practical, but their label models only handle classification.

Many tasks have richer labels. A recommender is trained on rankings, a pricing model on real numbers, a
taxonomy classifier on nodes of a tree. This library extends weak supervision to those settings. Each
labeling function is treated as a noisy copy of the true label in a metric space, and its accuracy is
estimated from how often it agrees with the other labeling functions. The label model then combines the
labeling functions with an accuracy-weighted average that suits the label space. For rankings that is a
weighted Kemeny ranking, and for real values it is a posterior mean.

The same recipe covers full and partial rankings, real-valued labels, any finite metric space, and
Riemannian manifolds such as hyperbolic space. The paper also proves finite-sample error bounds for the
accuracy estimates.

## Installation

```bash
pip install universalizing-weak-supervision
```

The PyPI distribution is `universalizing-weak-supervision`; the Python import remains `uws`.

Optional parts need extra dependencies:

```bash
pip install "universalizing-weak-supervision[l2r]"        # ranking end model (torch)
pip install "universalizing-weak-supervision[geodesic]"   # manifold-valued labels (torch, geoopt)
pip install "universalizing-weak-supervision[binary]"     # FlyingSquid binary label model
pip install "universalizing-weak-supervision[all]"        # everything, including the Snorkel baseline
```

To install from source and run the tests:

```bash
git clone https://github.com/SprocketLab/universalizing-weak-supervision
cd universalizing-weak-supervision
pip install -e ".[dev,all]"
pytest
```

`uws` needs Python 3.9 or newer.

## Sample Usage

This sample learns to rank sets of items from labeling functions that each output a ranking.

```python
"""
n = number of item sets (examples)
m = number of labeling functions
d = number of items per set
f = number of features per item

L_train: n lists of m rankings (uws.ranking.Ranking), or an (n, m, d) array of item orders
X_train, X_test: (n, d, f) arrays of item features, the end model's input
Y_test: true rankings of the test sets, for evaluation only
"""

from uws.label_model import RankingLabelModel
from uws.end_model import RankingEndModel
from uws.ranking import RankingUtils

# Learn labeling-function accuracies and produce training labels
label_model = RankingLabelModel(RankingUtils(d))
label_model.fit(L_train)
Y_train = label_model.predict(L_train)

# Train a ranking model on the generated labels
end_model = RankingEndModel(n_features=f)
end_model.fit(X_train, Y_train)

# Evaluate: mean normalized Kendall tau distance to the true rankings (lower is better)
score = end_model.score(X_test, Y_test, metric="kendall_tau")
```

`label_model.thetas` holds the estimated accuracy of each labeling function. By default the label model
uses the paper's estimator. If you have six or more labeling functions and some of them may be close to
random, `RankingLabelModel(..., train_method="sign_triplet_opt")` usually estimates their accuracies much
better. The [documentation](https://universalizing-weak-supervision.readthedocs.io) explains the
difference.

The other label models work the same way, with `fit(L)` and `predict(L)`:

| Labels | Label model in `uws.label_model` | What `L` contains |
|---|---|---|
| Full or partial rankings | `RankingLabelModel(RankingUtils(d))` | rankings, as above |
| Real values | `ContinuousLabelModel()` | an (n, m) float array; `fit` also takes the label variance |
| Any finite metric space | `MetricLabelModel(D)` with a distance matrix `D` | an (n, m) array of label indices, -1 to abstain |
| Points on a manifold | `GeodesicLabelModel(manifold)` | m tensors of shape (n, dim) |
| Binary | `BinaryLabelModel(m)`, a wrapper around FlyingSquid | an (n, m) array of votes in {-1, 0, +1} |

`uws.synthetic` generates labeling functions with known accuracies for every label type, which is a quick
way to try the models out.

## Tutorials

The notebooks in [`tutorials/`](tutorials) walk through the paper's experiments.

| Topic | Notebook |
|---|---|
| A first ranking example | `tutorials/synthetic/Ranking-Example.ipynb` |
| Rankings on real data (board games, movies) | `tutorials/{boardgames,movies}/RankingExperiments.ipynb` |
| Recovering rankings from synthetic labeling functions | `tutorials/synthetic/{Full,Partial}-Rankings-Experiments-Center-Recovery.ipynb` |
| Regression | `tutorials/synthetic/Regression-Experiments.ipynb`, `tutorials/{boardgames,movies}/RegressionExperiments.ipynb` |
| Hyperbolic (geodesic) regression | `tutorials/geodesic-regression/geodesic_regression.ipynb` |
| Generic finite metric spaces | `tutorials/metric-spaces/generic_metric_spaces.ipynb` |

## Reproducing the Paper

[`benchmarks/paper_ranking/`](benchmarks/paper_ranking) re-runs the paper's main ranking experiment
(Figure 2), which measures how a ranking model trained on weak labels improves as labeling functions are
added. The paper's comparison holds up: the UWS label model beats majority vote, which beats Snorkel, and
the numbers are close to the published ones. The fully supervised baselines come out somewhat stronger
than in the paper, so the claim that weak supervision beats training on 20% of the true labels only holds
for the newer `sign_triplet_opt` estimator with 15 or more labeling functions. The full tables are in
[`benchmarks/paper_ranking/summary_tables.md`](benchmarks/paper_ranking/summary_tables.md), and the other
scripts in [`benchmarks/`](benchmarks) check that legacy mode matches the original code.

To reproduce the original code's behavior, pass `legacy=True` to `RankingUtils`, `ContinuousLabelModel`
or `GeodesicLabelModel`. In the experiment configs, set `weak_sup_conf.legacy` and
`data_conf.legacy_sampling` to `true`.

## References

- Changho Shin, Winfred Li, Harit Vishwakarma, Nicholas Roberts, Frederic Sala.
  [Universalizing Weak Supervision](https://arxiv.org/abs/2112.03865). ICLR 2022.
- Alexander Ratner, Stephen Bach, Henry Ehrenberg, Jason Fries, Sen Wu, Christopher Ré.
  [Snorkel: Rapid Training Data Creation with Weak Supervision](https://arxiv.org/abs/1711.10160). VLDB 2018.
- Alexander Ratner, Braden Hancock, Jared Dunnmon, Frederic Sala, Shreyash Pandey, Christopher Ré.
  [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/abs/1810.02840). AAAI 2019.
- Daniel Fu, Mayee Chen, Frederic Sala, Sarah Hooper, Kayvon Fatahalian, Christopher Ré.
  [Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods](https://arxiv.org/abs/2002.11955). ICML 2020.
- Ronald Fagin, Ravi Kumar, Mohammad Mahdian, D. Sivakumar, Erik Vee.
  Comparing Partial Rankings. SIAM Journal on Discrete Mathematics, 2006.

## Release Notes

### Major changes in v1.0

- The research code is now an installable package with tests, documentation and continuous integration.
  Label models live in `uws.label_model`, end models in `uws.end_model`, and generators in `uws.synthetic`.
- Many bugs in the original code are fixed, and several approximations are replaced with exact methods.
  `kemeny` now finds the true Kemeny ranking, partial rankings use the Kendall distance of Fagin et al.,
  and Mallows sampling is exact. The paper's behavior remains available through `legacy=True`.
- The Gaussian label model no longer inverts ill-conditioned matrices, so correlated or numerous labeling
  functions no longer make it blow up.
- New: `MetricLabelModel` for any finite metric space, the `RankingEndModel` ranker, and the
  `sign_triplet_opt` estimator for rankings.

See [CHANGELOG.md](CHANGELOG.md) for the complete list.

## Developer Guidelines

Set up a development install with `pip install -e ".[dev,all]"` and `pre-commit install`. Before sending a
change, run `pytest` and `ruff check uws tests`. If a change alters numerical results, keep the old
behavior reachable through a `legacy` option and describe the change in `CHANGELOG.md`.
[CONTRIBUTING.md](CONTRIBUTING.md) has the details. To build the documentation, run
`pip install -e ".[docs]"` and then `sphinx-build docs docs/_build`.

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
