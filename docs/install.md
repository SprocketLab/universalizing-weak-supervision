# Installation

`uws` needs Python 3.9 or newer.

```bash
pip install universalizing-weak-supervision
```

The PyPI distribution is `universalizing-weak-supervision`; the Python import remains `uws`.

The core install covers rankings, regression and finite metric spaces. Other parts need extra
dependencies, which you can add with *extras*:

| Command | Adds |
|---|---|
| `pip install "universalizing-weak-supervision[geodesic]"` | manifold-valued labels (torch, geoopt) |
| `pip install "universalizing-weak-supervision[l2r]"` | learning-to-rank end models used in the paper's experiments (torch) |
| `pip install "universalizing-weak-supervision[binary]"` | the FlyingSquid binary label model |
| `pip install "universalizing-weak-supervision[snorkel]"` | the Snorkel baseline |
| `pip install "universalizing-weak-supervision[all]"` | all of the above |

## From source

```bash
git clone https://github.com/SprocketLab/universalizing-weak-supervision
cd universalizing-weak-supervision
pip install -e ".[dev,all]"
pytest
```
