# How it works

## The model

Labels live in a space with a distance $d$. For rankings this is the Kendall tau distance, for real values
the squared error, and for the nodes of a graph the shortest-path length. Each labeling function (LF) $j$
is modeled as a noisy copy of the true label $y$: an output becomes exponentially less likely the further
it is from $y$.

$$P(\lambda_j = x \mid y) \propto \exp(-\theta_j\, d(x, y))$$

The parameter $\theta_j$ is the LF's accuracy. A large $\theta_j$ means the LF is usually close to the
truth, and $\theta_j \approx 0$ means its outputs are essentially random. For rankings this is the Mallows
model, and for real values it is a Gaussian. The LFs are assumed to make their errors independently of
each other, given $y$.

A label model works in two stages. It first estimates every $\theta_j$ from unlabeled data, and then
combines the LF outputs using those estimates.

## Estimating accuracies

The true label is never observed, but the outputs of all LFs on the same example are. If two LFs often
agree, both are probably accurate. To make this precise, `uws` uses triplets of LFs $a, b, c$. The average
distance between each pair of them can be measured, and these three numbers determine the three unknown
accuracies.

The paper's estimator assumes that distances add up:

$$\mathbb{E}\, d(\lambda_a, \lambda_b) \approx \mathbb{E}\, d(\lambda_a, y) + \mathbb{E}\, d(\lambda_b, y).$$

Solving the three equations gives
$\mathbb{E}\, d(\lambda_a, y) = \tfrac12(\mathbb{E}\, d_{ab} + \mathbb{E}\, d_{ac} - \mathbb{E}\, d_{bc})$.
The `median_triplet` method takes the median of this estimate over every triplet that contains $a$, and
`triplet` uses disjoint triplets as the paper did. The expected distance is then converted to $\theta_j$.
The `*_opt` methods do this exactly by inverting the model's expected distance; the others use
$1/\mathbb{E}\,d$.

The additive assumption is exact for squared Euclidean distance. For other distances it is an
approximation, and for rankings it can be a poor one.

### Two estimators for rankings

Kendall distance counts the item pairs that two rankings order differently. Two independent LFs disagree
on a pair when exactly one of them gets it wrong, so their disagreement rates do not simply add. In the
extreme case, two random rankings disagree on half the pairs, and the additive rule concludes that each of
them is wrong on only a quarter. Near-random LFs therefore look far better than they are and keep almost
as much weight as good ones.

The `sign_triplet` and `sign_triplet_opt` methods use an exact relation instead. For a pair of items, let
$z = +1$ if an LF ranks the first item higher and $z = -1$ otherwise. Let $s_a = \mathbb{E}[z_a z_y]$ measure
how often LF $a$ agrees with the truth on a pair, where 1 means always and 0 means at chance. Independence
gives

$$\mathbb{E}[z_a z_b] = s_a\, s_b,$$

so the matrix of agreements between LFs has rank one. `uws` fits this matrix by least squares over all LF
pairs, weighting each pair by the number of item pairs both LFs ranked, which also handles partial
rankings. Each LF's expected Kendall distance is then $\#\text{pairs} \cdot (1 - s_a)/2$. Triplet methods for
binary labels, such as FlyingSquid, rest on the same idea; here it is applied to every item pair of a
ranking.

| | `median_triplet_opt`, `triplet_opt` | `sign_triplet_opt` |
|---|---|---|
| Relation used | distances add up (approximate for rankings) | pair agreements multiply (exact) |
| Near-random LFs | rated too highly, keep their weight | detected, get the smallest weight |
| Requirements | at least 3 LFs | at least 3 LFs that are better than random |
| Default | yes, this is the paper's estimator | no |

The figure compares the estimators on the LFs of the paper's Figure 2 experiment (see {doc}`reproducing`).
The top row plots estimated against true $\theta$, and the bottom row plots each LF's estimated distance to
the truth against its actual distance. The additive estimators assign every LF about the same accuracy.
The pair-sign fit follows the truth, except in the runs where only one or two LFs are informative, which
are the runs with 3 and 6 LFs.

```{image} _static/theta_recovery.png
:alt: estimated against true LF accuracy for the additive triplet, median triplet and pair-sign estimators
```

As a rule of thumb, use `sign_triplet_opt` when you have six or more LFs and some of them may be weak.
With very few LFs, use the paper's estimator.

## Combining labeling functions

The label model predicts the label that is closest to the LF outputs on average, with each LF weighted by
its accuracy. This is the weighted Fréchet mean:

$$\hat y = \arg\min_y \sum_j \theta_j\, d(\lambda_j, y).$$

- **Rankings.** This is the weighted Kemeny ranking, `inference_rule="weighted_kemeny"`. It is computed
  exactly for up to 12 items and by local search for more. `weighted_pairwise_majority` and
  `weighted_position_estimation` are cheaper alternatives, and the same names without `weighted_` ignore
  the accuracies.
- **Real values.** The prediction is the Gaussian posterior mean. `predict(L, return_var=True)` also
  returns the posterior variance.
- **Finite metric spaces.** The prediction is the best label among all candidates, and `predict_proba`
  gives a distribution over them.
- **Manifolds.** The prediction is the Riemannian Fréchet mean, found by gradient descent.

## Assumptions

- **Independent errors.** When several LFs are near-copies of each other, such as one heuristic at
  different thresholds, triplets that combine them look too consistent and those LFs get too much weight.
  The ranking and metric label models accept `groups` in `fit`, one family label per LF, and then only build
  triplets across families.
- **Better than random.** Accuracies are only identified up to a sign, so `uws` assumes that no LF is
  systematically wrong. An LF that reliably reverses rankings would be misread.

## Legacy mode

Passing `legacy=True` to `RankingUtils`, `ContinuousLabelModel` or `GeodesicLabelModel`, and
`sampler="legacy"` to `Mallows`, reproduces the original paper code exactly, including the approximations
that version 1.0 corrects. The {doc}`changelog` lists what changed.
