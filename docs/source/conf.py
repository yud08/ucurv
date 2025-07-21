# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) 

project = 'UDCT'
copyright = '2025, Duy Nguyen'
author = 'Duy Nguyen'
release = '0.1.1'

extensions = [
    "sphinx.ext.autodoc",    
    "sphinx.ext.napoleon",       # Google/NumPy‑style docstrings
    "sphinx.ext.autosectionlabel"  # optional
]

autosectionlabel_prefix_document = True   
html_theme = 'sphinx_rtd_theme'               


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
