# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import sys
import os

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bhepop2"
copyright = "2023, uge-tellae"
author = "uge-tellae"

# Get the version from the __init__ module
version_module = importlib.import_module("bhepop2.__init__")
version = ".".join(version_module.__version__.split(".")[0:2])

# The full version, including alpha/beta/rc tags
release = version_module.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
]

# autdoc config value
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

# The master toctree document.
master_doc = "contents"

# package directory to document with autoapi
autoapi_dirs = ["../bhepop2"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

todo_include_todos = True
