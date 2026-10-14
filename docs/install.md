# Installation

```bash
pip install uws                 # core
pip install "uws[geodesic]"     # manifold-valued labels (torch, geoopt)
pip install "uws[l2r]"          # learning-to-rank end models (torch)
pip install "uws[binary]"       # FlyingSquid binary label model
pip install "uws[all]"          # everything incl. the Snorkel baseline
```

Development install: `pip install -e ".[dev,all]"`.
