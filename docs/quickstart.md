# Quickstart

Every label model in `uws` works the same way:

1. Collect the outputs of your labeling functions (LFs) on unlabeled data into `L`.
2. `fit(L)` estimates how accurate each LF is. No labels are needed.
3. `predict(L)` combines the LF outputs into one label per example, trusting accurate LFs more.

## Rankings

Six LFs rank the same 5 items in 1000 examples. Three of them are essentially random.

```python
import numpy as np
from uws.ranking import RankingLabelModel, RankingUtils, generate_true_rankings, sample_mallows_LFs

rng = np.random.default_rng(0)
r_utils = RankingUtils(d=5)                         # rankings of 5 items
Y = generate_true_rankings(5, 1000, rng, r_utils)   # true rankings, only used for evaluation
L = sample_mallows_LFs(Y, 6, [0.01, 0.01, 0.01, 0.5, 1.0, 2.0], rng=rng)

lm = RankingLabelModel(r_utils, train_method="sign_triplet_opt").fit(L)
Y_hat = lm.predict(L)

print(lm.thetas)                            # [0.1  0.1  0.1  0.55 1.08 1.88]
print(r_utils.mean_kt_distance(Y_hat, Y))   # 0.061 (lower is better)
```

`lm.thetas` are the estimated LF accuracies: the three random LFs get the lowest possible weight.
Without weights (`lm.predict(L, inference_rule="kemeny")`) the error is 0.186.

`L` can be a list with one entry per example, each a list with one {class}`uws.ranking.Ranking` per LF, or an
integer array of shape `(n_examples, n_LFs, n_items)`. For partial rankings, hide items with
`Ranking.mask_items`.

## Regression

```python
from uws.regression import ContinuousLabelModel, generate_lfs, generate_synthetic_xy

X, Y, var_Y = generate_synthetic_xy(5000, 10)    # synthetic data with label variance var_Y
L, _ = generate_lfs(8, 5000, Y, var_Y)           # 8 noisy real-valued LFs, shape (5000, 8)

lm = ContinuousLabelModel().fit(L, var_Y)
y_hat, posterior_var = lm.predict(L, return_var=True)
```

You need to supply the variance of the label (`var_Y`), for example from domain knowledge or a small
labeled sample. The mean of the label is estimated from the LFs unless you pass `mean_Y`.

## Any finite metric space

Labels can be any finite set with a distance between its elements, for example the nodes of a graph with
shortest-path distances:

```python
from uws.metric import MetricLabelModel

D = ...                            # (k, k) distance matrix between the k possible labels
lm = MetricLabelModel(D).fit(L)    # L: (n, m) label indices, -1 where an LF abstains
y_hat = lm.predict(L)
proba = lm.predict_proba(L)        # (n, k) distribution over labels
```

## Command line

The paper's real-data ranking pipeline (generate LFs, fit the label model, train a learning-to-rank model)
runs from a config file:

```bash
uws-experiment configs/boardgames_ranking_experiment.yaml --project-root . \
    --set weak_sup_conf.num_LFs=6 --train
```

`--set key=value` overrides any config entry; without `--train` only the label model step runs.
