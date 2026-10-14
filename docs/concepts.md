# How it works

Every UWS label model assumes LF $j$ follows an exponential family around the latent label $y$ in a metric
space $(\mathcal{Y}, d)$:

$$P(\lambda_j = x \mid y) \propto \exp(-\theta_j\, d(x, y)).$$

**Parameter estimation (no labels).** For conditionally independent LFs,
$\mathbb{E}[d(\lambda_a, \lambda_b)] \approx \mathbb{E}[d(\lambda_a, y)] + \mathbb{E}[d(\lambda_b, y)]$, so the
per-LF expected distances follow from LF-LF distances through triplets $(a, b, c)$:
$\mathbb{E}[d(\lambda_a, y)] = \tfrac12(\mathbb{E} d_{ab} + \mathbb{E} d_{ac} - \mathbb{E} d_{bc})$
({func}`uws.utils.individual_distances`). The expected distance is then mapped to $\theta_j$: exactly for Mallows
models (`*_opt` train methods), or as $\theta_j = 1 / \mathbb{E}[d]$.

**Inference.** The weighted Fréchet mean $\hat y = \arg\min_y \sum_j \theta_j\, d(\lambda_j, y)^q$. For rankings with
Kendall's tau this is the weighted Kemeny rule; for the Gaussian model it is the posterior mean.

**Legacy mode.** `legacy=True` (`RankingUtils`, `ContinuousLabelModel`, `GeodesicLabelModel`) and
`Mallows(..., sampler="legacy")` reproduce the numbers of the original paper code. See the changelog for what
changed.
