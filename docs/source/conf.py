# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Make project importable
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "HECATE"
copyright = "2026, Telmo Monteiro"
author = "Telmo Monteiro"
release = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
]

autosummary_generate = True

add_module_names = False
autodoc_class_signature = "separated"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = True

# Mock heavy imports so docs build fast
autodoc_mock_imports = [
    "SOAP",
    "matplotlib",
    "dynesty",
    "ldtk",
    "numpy",
    "scipy",
    "astropy",
]

# templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Section labels usable across files
autosectionlabel_prefix_document = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Only include if folder exists
html_static_path = ["_static"]

# nicer sidebar depth
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# copybutton config
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True