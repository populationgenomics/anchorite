import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

project = 'anchorite'
copyright = '2026, Centre for Population Genomics'
author = 'Tobias Sargeant'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser',
]

# Slug ids for h1-h3, so the README's in-page links (#normalisation) resolve.
myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns: list[str] = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
