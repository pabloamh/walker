# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Walker'
copyright = '2024, Pablo'
author = 'Pablo'

# -- Path setup --------------------------------------------------------------

# Add the project's source directory to the path
# This is necessary for Sphinx to find and import your modules for autodoc
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'myst_parser',             # Parse Markdown files like README.md
    'sphinx_click',            # Document Click-based CLI applications
]

templates_path = ['_templates']
exclude_patterns = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = 'bysource'

# -- MyST Parser settings ----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
]

# -- Sphinx-Click settings ---------------------------------------------------
# This tells sphinx-click where to find your Click application object.
sphinx_click_mod_path = 'walker.main'
sphinx_click_obj = 'cli'
sphinx_click_show_hidden = False
