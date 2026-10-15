# Installation

`uws` needs Python 3.9 or newer.

```bash
pip install uws
```

The core install covers rankings, regression and finite metric spaces. Other parts need extra
dependencies, which you can add with *extras*:

| Command | Adds |
|---|---|
| `pip install "uws[geodesic]"` | manifold-valued labels (torch, geoopt) |
| `pip install "uws[l2r]"` | learning-to-rank end models used in the paper's experiments (torch) |
| `pip install "uws[binary]"` | the FlyingSquid binary label model |
| `pip install "uws[snorkel]"` | the Snorkel baseline |
| `pip install "uws[all]"` | all of the above |

## From source

```bash
git clone https://github.com/SprocketLab/universalizing-weak-supervision
cd universalizing-weak-supervision
pip install -e ".[dev,all]"
pytest
```
