# Universalizing Weak Supervision

`uws` is a weak supervision library for labels that are not just classes: rankings, real numbers, nodes
of a graph, points in hyperbolic space, and more. It accompanies the ICLR 2022 paper
[Universalizing Weak Supervision](https://arxiv.org/abs/2112.03865) by Changho Shin, Winfred Li,
Harit Vishwakarma, Nicholas Roberts and Frederic Sala.

Weak supervision replaces hand-labeled training data with labeling functions: heuristics, existing
models, or crowd workers that each produce noisy labels. A label model learns how accurate each labeling
function is, without seeing any ground truth, and combines their outputs into training labels. Label models
in tools such as Snorkel only handle classification. The label models in `uws` work in any metric space,
so the same approach applies to rankings, regression and structured outputs.

The library has four main parts:

- {mod}`uws.label_model`: the label models, one per kind of label.
- {mod}`uws.end_model`: models trained on the labels the label model produces.
- {mod}`uws.ranking`: the ranking type, Kendall tau distances, Mallows models and rank aggregation.
- {mod}`uws.synthetic`: labeling functions with known accuracies, for experiments and tests.

Start with {doc}`quickstart`, then read {doc}`concepts` to see how the label models work and which
options to choose.

```{toctree}
:maxdepth: 2

install
quickstart
concepts
reproducing
api
changelog
```
