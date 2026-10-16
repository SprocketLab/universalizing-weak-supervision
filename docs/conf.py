import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from uws.version import __version__  # noqa: E402

project = "Universalizing Weak Supervision"
author = "Changho Shin, Winfred Li, Harit Vishwakarma, Nicholas Roberts, Frederic Sala"
copyright = "2021-2026, " + author
release = version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]
autosummary_generate = True
autosummary_imported_members = True
autodoc_mock_imports = ["torch", "geoopt", "flyingsquid", "snorkel"]
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
napoleon_numpy_docstring = True
myst_enable_extensions = ["dollarmath"]
html_theme = "furo"
html_title = f"uws {version}"
exclude_patterns = ["_build"]
