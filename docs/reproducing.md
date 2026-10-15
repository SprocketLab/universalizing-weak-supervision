# Reproducing the paper

## Notebooks

The notebooks in `tutorials/` walk through the paper's experiments:

| Experiment | Notebook |
|---|---|
| Rankings on real data | `tutorials/{boardgames,movies}/RankingExperiments.ipynb` |
| Synthetic rankings | `tutorials/synthetic/Ranking-Example.ipynb` (shortest), `Full-` and `Partial-Rankings-Experiments-Center-Recovery.ipynb` |
| Regression | `tutorials/{boardgames,movies}/RegressionExperiments.ipynb`, `tutorials/synthetic/Regression-Experiments.ipynb` |
| Hyperbolic regression | `tutorials/geodesic-regression/geodesic_regression.ipynb` |
| Generic metric spaces | `tutorials/metric-spaces/generic_metric_spaces.ipynb` |

The ranking notebooks contain a cell that switches on legacy mode (`weak_sup_conf['legacy'] = True` and
`data_conf['legacy_sampling'] = True`), which reproduces the original code's behavior, including how it
sampled the item sets.

## Figure 2: real-data rankings

`benchmarks/paper_ranking/reproduce_figure2.py` re-runs the paper's main ranking experiment: 5000 training
and 1000 test sets of 5 board games or movies, simulated LFs of which one third are good and the rest nearly
random, and a ListMLE ranking model trained on the aggregated labels. The score is the Kendall tau distance
of the trained model on the test set (lower is better). All methods aggregate the same LF outputs. Three
seeds per setting; the paper's values are read off its Figure 2.

Board games, test distance by number of LFs:

| Method | 3 | 6 | 9 | 12 | 15 | 18 |
|---|---:|---:|---:|---:|---:|---:|
| Paper: UWS | 0.221 | 0.200 | 0.195 | 0.184 | 0.187 | 0.186 |
| Rerun: UWS, legacy mode | 0.236 | 0.211 | 0.191 | 0.187 | 0.187 | 0.181 |
| Rerun: majority vote, legacy mode | 0.243 | 0.224 | 0.204 | 0.197 | 0.192 | 0.190 |
| Rerun: Snorkel | 0.254 | 0.262 | 0.223 | 0.209 | 0.219 | 0.211 |
| Rerun: UWS 1.0, paper estimator | 0.232 | 0.206 | 0.197 | 0.188 | 0.185 | 0.181 |
| Rerun: UWS 1.0, `sign_triplet_opt` | 0.285 | 0.191 | 0.178 | 0.176 | 0.166 | 0.171 |

Fully supervised, 20% / 50% / 100% of the training labels: paper 0.184 / 0.177 / 0.172, rerun
0.175 / 0.163 / 0.160.

What this shows:

* **The paper's comparison reproduces.** In legacy mode, UWS beats majority vote, which beats Snorkel, at
  every number of LFs. On board games UWS is within 0.015 of the paper's numbers; movies are noisier (up to
  0.04 apart at 6 LFs).
* **The supervised baselines do not.** They come out 0.01 to 0.025 better than in the paper, perhaps because
  of newer PyTorch or training settings the paper does not report. Against these baselines, the paper's estimator never
  reaches the 20% baseline (best 0.181 vs 0.175).
* **The pair-sign estimator gets closer.** From 6 LFs on, `sign_triplet_opt` is the best weakly supervised
  method. It beats the 20% baseline at 15 and 18 LFs on board games (0.166 at 15 LFs, between the 20% and
  50% baselines) and at 15 LFs on movies. With 3 LFs (only one good) it is worse and unstable.
* **Exact Kemeny vs the paper's rule makes little difference**: UWS 1.0 with the paper's estimator performs
  like legacy mode. The gain comes from the accuracy estimates.

The movies results show the same pattern and are in `benchmarks/paper_ranking/summary_tables.md`, along with
standard deviations and the quality of the aggregated training labels.

`benchmarks/paper_ranking/theta_recovery.py` re-fits both estimators on the saved LF outputs of this sweep
and produces the accuracy-recovery figure in {doc}`concepts`.

### Running it

One job is one dataset, seed and number of LFs (`--m 0` trains the fully supervised baselines):

```bash
python benchmarks/paper_ranking/reproduce_figure2.py --dataset boardgames --seed 0 --m 6 --out results/
```

A job with 18 LFs takes about 40 minutes on a CPU node (five methods, each training the ranking model).

## Differences from the paper's setup

* The paper states that good LFs have $\theta \sim U(0.2, 1)$; the shipped configs and the original code's
  defaults use $U(2, 5)$. The reproduction above uses the paper's range.
* Exact numbers from the original notebooks cannot be regenerated: the original code drew part of its
  randomness from Python's unseeded `random` module. Reruns of the shipped notebook configurations in legacy
  mode fall within 1.5 standard deviations of the saved outputs.
