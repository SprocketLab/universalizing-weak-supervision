# Quickstart

Every label model in `uws` is used in three steps:

1. Collect the outputs of your labeling functions on unlabeled data in `L`.
2. Call `fit(L)` to estimate how accurate each labeling function is. No labels are needed.
3. Call `predict(L)` to combine the outputs into one label per example. Accurate labeling functions count
   for more.

The resulting labels can then train an end model.

## Rankings

The example below creates sets of 5 items, each item described by 3 features. The true ranking of each
set depends on the features, but neither the label model nor the end model ever sees it. Six labeling
functions rank every set, and three of them are close to random.

```python
import numpy as np
from uws.end_model import RankingEndModel
from uws.label_model import RankingLabelModel
from uws.ranking import Ranking, RankingUtils
from uws.synthetic import sample_mallows_LFs

rng = np.random.default_rng(0)
n, d, f = 2000, 5, 3
r_utils = RankingUtils(d)

# Item features and the true rankings they imply
X = rng.normal(size=(n, d, f))
true_scores = X @ np.array([1.0, -0.5, 0.2]) + 0.3 * rng.normal(size=(n, d))
Y = [Ranking(np.argsort(-s), r_utils) for s in true_scores]

# Six labeling functions; the first three are close to random
L = sample_mallows_LFs(Y, 6, [0.01, 0.01, 0.01, 0.5, 1.0, 2.0], rng=rng)
L_train, X_train = L[:1500], X[:1500]
X_test, Y_test = X[1500:], Y[1500:]

# Label model: estimate accuracies, then aggregate into training labels
label_model = RankingLabelModel(r_utils, train_method="sign_triplet_opt")
label_model.fit(L_train)
Y_train = label_model.predict(L_train)
print(label_model.thetas.round(2))   # [0.1  0.1  0.1  0.51 1.1  2.02]

# End model: learn to rank from features using the generated labels
end_model = RankingEndModel(n_features=f, seed=0)
end_model.fit(X_train, Y_train)
print(end_model.score(X_test, Y_test))   # 0.115
```

The label model gives the three near-random labeling functions the smallest possible weight. The score is
the mean normalized Kendall tau distance between predicted and true rankings: 0 means identical rankings
and 0.5 is what random rankings get.

`L` can be a list with one entry per example, each a list with one {class}`~uws.ranking.Ranking` per
labeling function, or an integer array of shape `(n_examples, n_LFs, n_items)`. For partial rankings, hide
items with `Ranking.mask_items`; the end model ignores hidden items when training.

## Real values

```python
from uws.label_model import ContinuousLabelModel
from uws.synthetic import generate_lfs, generate_synthetic_xy

X, Y, var_Y = generate_synthetic_xy(5000, 10)
L, _ = generate_lfs(8, 5000, Y, var_Y)   # 8 noisy real-valued labeling functions, shape (5000, 8)

label_model = ContinuousLabelModel().fit(L, var_Y)
y_hat, posterior_var = label_model.predict(L, return_var=True)
```

The model needs the variance of the label, `var_Y`, which can come from domain knowledge or a small
labeled sample. The mean of the label is estimated from the labeling functions unless you pass `mean_Y`.

## Any finite metric space

Labels can come from any finite set with a distance between its elements, such as the nodes of a graph
with shortest-path distances.

```python
from uws.label_model import MetricLabelModel

D = ...   # (k, k) distance matrix between the k possible labels
label_model = MetricLabelModel(D).fit(L)   # L: (n, m) label indices, -1 where a labeling function abstains
y_hat = label_model.predict(L)
proba = label_model.predict_proba(L)       # (n, k) distribution over labels
```

## Command line

The paper's real-data ranking pipeline runs from a config file:

```bash
uws-experiment configs/boardgames_ranking_experiment.yaml --project-root . \
    --set weak_sup_conf.num_LFs=6 --train
```

`--set key=value` overrides any config entry. Without `--train`, only the label model step runs.
