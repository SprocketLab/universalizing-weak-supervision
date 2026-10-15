# How it works

## The model

Labels live in a space with a distance $d$: Kendall tau distance for rankings, squared error for real
values, shortest-path length for nodes of a graph, and so on. Each labeling function $j$ is modeled as a
noisy copy of the true label $y$ whose outputs become exponentially less likely the further they are from $y$:

$$P(\lambda_j = x \mid y) \propto \exp(-\theta_j\, d(x, y)).$$

The parameter $\theta_j$ is the LF's accuracy: large $\theta_j$ means the LF is usually close to the truth,
$\theta_j \approx 0$ means its outputs are essentially random. For rankings this is the Mallows model; for
real values it is a Gaussian. LFs are assumed to make their errors independently of each other, given $y$.

Using `uws` has two steps: estimate every $\theta_j$ from unlabeled data, then combine the LF outputs.

## Step 1: estimating accuracies without labels

The true label is never observed, but the LFs' outputs on the same example are. If two LFs often agree,
both are probably accurate. To turn this into numbers, `uws` uses *triplets*: for three LFs $a, b, c$, the
average distances between each pair, $\mathbb{E}\,d(\lambda_a, \lambda_b)$ and so on, are three observable
quantities that pin down the three unknown accuracies.

The paper's estimator assumes that distances add up:

$$\mathbb{E}\, d(\lambda_a, \lambda_b) \approx \mathbb{E}\, d(\lambda_a, y) + \mathbb{E}\, d(\lambda_b, y),$$

which gives $\mathbb{E}\, d(\lambda_a, y) = \tfrac12(\mathbb{E}\, d_{ab} + \mathbb{E}\, d_{ac} - \mathbb{E}\, d_{bc})$.
`median_triplet` takes the median of this over every triplet that contains $a$; `triplet` uses disjoint
triplets as in the paper. The expected distance is then converted to $\theta_j$: exactly with the `*_opt`
methods (by inverting the model's expected distance), otherwise as $1/\mathbb{E}\,d$.

The additive assumption is exact for squared Euclidean distance. For other distances it is an approximation,
and for rankings it can be a poor one.

### Two estimators for rankings

Kendall distance counts the item pairs that two rankings order differently. Two independent LFs disagree on
a pair if exactly one of them gets it wrong, so the disagreement rates do not simply add. In the extreme
case, two random rankings disagree on half the pairs, and the additive rule concludes that each is wrong on
only a quarter of them. Near-random LFs are therefore judged far better than they are and keep almost as
much weight as good LFs.

`sign_triplet` and `sign_triplet_opt` use the exact relation instead. Write $z = \pm 1$ for whether an LF
puts item $i$ above item $j$, and $s_a = \mathbb{E}[z_a z_y]$ for how often LF $a$ agrees with the truth on a
pair (1 = always, 0 = random). Independence gives

$$\mathbb{E}[z_a z_b] = s_a\, s_b,$$

so the matrix of LF-LF agreements is rank one. `uws` fits it by least squares over all LF pairs, weighted by
how many item pairs both LFs observed (which handles partial rankings), and reads off each LF's expected
Kendall distance as $\#\text{pairs} \cdot (1 - s_a)/2$. This is the same idea that triplet methods for binary
labels use (e.g. FlyingSquid), applied to every item pair of a ranking.

| | `median_triplet_opt` / `triplet_opt` | `sign_triplet_opt` |
|---|---|---|
| Identity used | additive distances (approximate for rankings) | product of pair agreements (exact per pair) |
| Near-random LFs | overestimated, keep weight | detected, get the minimum weight |
| Needs | 3 LFs | at least 3 LFs that are better than random |
| Default | yes (the paper's estimator) | no |

The figure shows both on the LFs of the paper's Figure 2 experiment ({doc}`reproducing`): estimated vs true
θ (top) and estimated vs actual distance to the truth (bottom). The additive estimators place every LF at
about the same accuracy. The pair-sign fit tracks the truth except when only one or two LFs are informative
(3 and 6 LFs in that experiment).

```{image} _static/theta_recovery.png
:alt: estimated vs true LF accuracy for the additive triplet, median triplet and pair-sign estimators
```

A rule of thumb: use `sign_triplet_opt` when you have six or more LFs and some may be weak; use the paper's
estimator with very few LFs.

## Step 2: combining LF outputs

The label model predicts the point that is closest to the LF outputs on average, weighted by accuracy (the
*weighted Fréchet mean*):

$$\hat y = \arg\min_y \sum_j \theta_j\, d(\lambda_j, y).$$

* **Rankings:** this is the weighted Kemeny ranking (`inference_rule="weighted_kemeny"`), computed exactly
  for up to 12 items and by local search beyond. Cheaper alternatives are `weighted_pairwise_majority` and
  `weighted_position_estimation`; dropping `weighted_` ignores the accuracies.
* **Real values:** the Gaussian posterior mean, with the posterior variance available from
  `predict(L, return_var=True)`.
* **Finite metric spaces:** the minimizer over all labels, plus `predict_proba` for a distribution.
* **Manifolds:** the Riemannian Fréchet mean, found by gradient descent.

## Assumptions to keep in mind

* **Independent errors.** If several LFs are near-copies of each other (the same heuristic with different
  thresholds, the same model family), triplets that combine them look overly consistent and those LFs get
  too much weight. Passing `groups` (one family label per LF) to `fit` of the ranking or metric label model builds triplets only
  across families.
* **Better than random.** Accuracies are estimated up to a sign; `uws` assumes LFs are not systematically
  wrong (no LF that reliably reverses rankings).

## Legacy mode

`legacy=True` (on `RankingUtils`, `ContinuousLabelModel` and `GeodesicLabelModel`) and `Mallows(...,
sampler="legacy")` reproduce the original paper code exactly, including the approximations corrected in
version 1.0. See {doc}`changelog` for what changed.
