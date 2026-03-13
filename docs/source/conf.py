import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "anchorite"
copyright = "2026, Centre for Population Genomics"  # noqa: A001
author = "Tobias Sargeant"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
