# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../torchdr"))

from torchdr.__about__ import __version__  # Import the version from __about__.py

project = "TorchDR"
copyright = "2024, TorchDR team"
author = "Hugues Van Assel"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

autosummary_generate = True

copybutton_exclude = ".linenos, .gp"

intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://docs.pytorch.org/2.0/", None),
    "python": ("https://docs.python.org/3.4", None),
    "pykeops": ("https://www.kernel-operations.io/keops/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/"],
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/demo_",
    "run_stale_examples": True,
    "ignore_pattern": r"__init__\.py",
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("torchdr"),
    # objects to exclude from implicit backreferences. The default option
    # is an empty set, i.e. exclude nothing.
    "exclude_implicit_doc": {},
    "nested_sections": False,
}

# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {"equationNumbers": {"autoNumber": "AMS", "useLabelIds": True}}
}

math_numfig = True
numfig = True
numfig_secnum_depth = 3

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = []
html_favicon = "figures/logo.ico"
html_logo = "figures/torchdr_logo.png"
html_theme_options = {
    # "analytics_id": "",  # Provided by Google in your dashboard G-
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TorchDR/TorchDR",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/torchdr/",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ],
    "analytics_anonymize_ip": False,
    # Toc options
    "collapse_navigation": True,
    "navigation_depth": 4,
}

html_static_path = ["js"]
html_js_files = [
    "pypi-icon.js",
]

# Separator substitution : Writing |sep| in the rst file will display a horizontal line.
rst_prolog = """
.. |sep| raw:: html

   <hr />
"""
