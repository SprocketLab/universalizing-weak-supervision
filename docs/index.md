# Universalizing Weak Supervision

Weak supervision trains models from noisy, cheap label sources (heuristics, other models, crowd workers)
instead of hand labels. Most weak supervision tools only handle classification. `uws` handles **any label
type**: rankings, real numbers, points in a graph or any other finite metric space, and points on a manifold.

It is the library for the ICLR 2022 paper
[Universalizing Weak Supervision](https://arxiv.org/abs/2112.03865) by Changho Shin, Winfred Li,
Harit Vishwakarma, Nicholas Roberts and Frederic Sala.

New here? Start with {doc}`quickstart`, then read {doc}`concepts` for how the label models work and which
options to pick.

```{toctree}
:maxdepth: 2

install
quickstart
concepts
reproducing
api
changelog
```
