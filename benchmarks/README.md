# Benchmarks

Scripts that check the library against the original paper code and measure the effect of the 1.0 changes.
They are not part of the package or the test suite.

| Script | What it checks |
|---|---|
| `legacy_equivalence.py` | `legacy=True` gives the same numbers as the original research code on identical inputs. |
| `regression_stability.py` | The Gaussian label model on hard cases (near-duplicate LFs, mixed scales, many LFs), old vs new. |
| `geodesic_inference.py` | Fréchet-mean inference for the hyperbolic label model, old vs new. |
| `paper_ranking/reproduce_figure2.py` | Re-runs the paper's Figure 2 (real-data rankings); results in `paper_ranking/summary_tables.md`. |
| `paper_ranking/theta_recovery.py` | How well each ranking estimator recovers LF accuracies on the Figure 2 LFs (`paper_ranking/figures/`). |
| `jev_ws/` | Exploratory: UWS label models as aggregators for a pool of LLM judges (an external project). |

The documentation page "Reproducing the paper" summarizes the Figure 2 results.
