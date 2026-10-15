# Changelog

## 1.0.0 (2026-10)

First packaged release. The research code under `code/` and `uws/label_model.py` has been reorganized into
the installable `uws` package (`pip install uws`). Notebooks moved to `tutorials/` and import `uws`.

### Packaging
- `pyproject.toml`, optional extras (`l2r`, `geodesic`, `binary`, `snorkel`, `all`, `dev`, `docs`), Apache-2.0.
- Vendored Snorkel removed (it was an unmodified 0.9.6 copy); the baseline now uses `pip install snorkel`.
- The modified PT-Ranking fork is vendored as `uws._vendor.ptranking`.
- Tests (pytest), CI (GitHub Actions), Sphinx docs, PyPI release workflow, pre-commit, `uws-experiment` CLI.
- Library modules no longer seed global RNGs or configure logging at import; randomness goes through
  explicit `seed` / `rng` arguments.

### New
- `uws.metric.MetricLabelModel`: the label model for any finite metric space (previously notebook-only), with
  abstains, `predict_proba` and exact exponential-family theta inversion.
- Shared triplet estimators `uws.utils.individual_distances` (`triplet`, `median_triplet`, `mean_triplet`,
  `lstsq`, optional family-aware triplets via `groups`).
- A common `BaseLabelModel` interface (`fit`, `predict`, `get_weights`, `save`, `load`).
- `RankingLabelModel` (new name of `WeakSupRanking`, which remains as an alias) with `fit`/`predict`, typed
  `TrainMethod`/`InferenceRule` options and `lstsq` training.
- `sign_triplet` / `sign_triplet_opt` ranking train methods: estimate each LF's agreement with the latent
  ranking from the exact pair-sign identity `E[z_a z_b] = s_a s_b` (rank-1 fit over all LF pairs) instead of the
  additive distance identity, which underestimates near-random LFs because Kendall distances saturate. Masking
  does not bias it. On synthetic Mallows data it matches or beats `median_triplet_opt` everywhere tested
  (weighted-Kemeny KT 0.114 → 0.025 with four near-random LFs out of six; 0.184 → 0.125 for partial rankings with
  p = 0.5) and is close to inference with the true thetas. On the LFs of the paper's Figure 2 experiment it
  recovers LF accuracies far better (mean error in normalized Kendall distance 0.006 vs 0.24 for near-random LFs)
  and gives the best end model from 6 LFs on. It needs at least three better-than-random LFs. Not the default
  (the paper's estimator is); both are supported.
- `uws.geodesic`: `GeodesicRegressor`, `LFGenerator` and data generators from the geodesic notebook.

### Algorithmic corrections (`legacy=True` / `sampler="legacy"` restore the paper's behavior)
- **Kemeny**: `kemeny`/`weighted_kemeny` now compute the true (weighted) Kemeny ranking (exact dynamic program
  for d ≤ 12, Borda + insertion local search above). The previous rule returned the LF ranking with the smallest
  mean distance to the others; it is available as `kemeny_medoid`/`weighted_kemeny_medoid`.
- **Partial Kendall tau**: Fagin et al.'s K^(p) (penalty `p=0.5` for pairs ordered by one ranking only, 0 for
  pairs ordered by neither). Previously 0.1 and `1/#pairs`, respectively. Normalization no longer divides by zero.
- **Mallows sampling**: exact repeated insertion model for every d (the old sampler enumerated all d! permutations
  on every draw for d < 10 and used an approximate swap heuristic for d ≥ 10).
- **Gaussian label model**: no explicit matrix inverse. By default the model's covariance structure
  (rank-1 + diagonal) gives a closed-form, always-defined posterior, and `solver="empirical"` uses a shrunk sample
  covariance. Also: LF centering with a least-squares estimate of `mean_Y`; log-domain median triplets that skip
  degenerate triplets; accuracy signs recovered from the LF covariance; Cauchy-Schwarz clipping;
  `predict(..., return_var=True)`. On synthetic data with near-duplicate, mixed-scale LFs the median test MSE
  drops from 1.49 (max 13.1, 2/20 runs NaN) to 0.10 (oracle ≈ 0.09).
- **Geodesic label model**: uses all LFs and every triplet. The Fréchet mean is computed for all examples
  at once by Riemannian gradient descent with a float64 line search, instead of a fixed step that had to match
  the scale of theta.

### Bug fixes
- `GeodesicLabelModel` referenced undefined `man` and `tqdm`.
- Gaussian label model: `predict` looped over the *training* size instead of `len(L)`; `random.sample(set)`
  crashes on Python ≥ 3.11; `score` ignored `metric`; NaN accuracies for negative covariance ratios.
- `RankingDiGraph(ranking)` crashed (wrong `add_edge` call); `mask_node` built invalid edges.
- Ranking label model: LFs beyond the last full triplet (`m % 3 != 0`) got expected distance 0 and theta = inf;
  non-positive estimates gave negative/inf thetas; unknown `train_method`/`inference_rule` silently did nothing;
  medoid inference returned aliases of LF rankings which were then mutated.
- `Ranking.mask_items` shuffled the caller's list in place; `mask_permutation` had the wrong shape.
- `apply_masks(d_mask=k)` masked the *last* `k` items instead of keeping the top `k`.
- `restore_ranking` used an un-imported `Ranking`.
- Synthetic classification workflow: called a non-existent `inferRanking`, used `.shape` on lists, and
  `build_dict_d_theta_mu` never stored its results.
- Board-games dataset: reloading cached samples raised `KeyError` (`lst_movie_id_*` vs `lst_id_*`); train and test
  item sets were drawn with the same reseeded RNG stream.
- `DiscreteLF` failed on non-numeric columns with pandas ≥ 2; `ContinuousLF` divided by a zero quantile.
- `BasicColumnarDataset`: `drop_id_feature` was ignored; the id index was built in O(n²).
- `SyntheticRankingDataset` constructed but never raised its `AssertionError`s; `create_dataset` raised
  `NotImplemented` (not an exception).
- `Mallows.find_eq_theta_to_p0` referenced an undefined `mlw` in its Jacobian.
- `PtrankingWrapper` passed the un-rooted result path to PT-Ranking.
- Notebooks: NumPy 2 / pandas 2 / matplotlib incompatibilities (`DataFrame.append`, array-to-scalar assignment,
  `markerSize`).
