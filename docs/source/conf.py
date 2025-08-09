# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) 

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UDCT'
copyright = '2025, Duy Nguyen'
author = 'Duy Nguyen'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",          # pull in docstrings
    "sphinx.ext.napoleon",         # Google/NumPy-style docstrings
    "sphinx.ext.autosectionlabel", # reference sections across docs
    "sphinx.ext.viewcode",         # link to highlighted source
    "sphinx.ext.autosummary"       # auto-generate API summary pages
]

autosectionlabel_prefix_document = True   
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ['_templates']
exclude_patterns = []

# If your code imports heavy optional deps, mock them to avoid build failures
autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "pylops"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # readthedocs theme for API docs
html_static_path = ['_static']
