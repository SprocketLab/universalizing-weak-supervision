# Contributing

1. `pip install -e ".[dev,all]"` and `pre-commit install`.
2. Add tests under `tests/` for new behavior; `pytest` must pass.
3. `ruff check uws tests` and `ruff format` (line length 120). `uws/_vendor` is third-party code: keep changes
   there minimal and documented.
4. Changes that alter numerical results must keep the paper's behavior reachable (a `legacy` flag) and be noted
   in `CHANGELOG.md`.

Releases: bump `uws/version.py`, update the changelog, tag `vX.Y.Z`; the release workflow builds and publishes to
PyPI through trusted publishing.
