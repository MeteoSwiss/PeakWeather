import os
import sys
sys.path.insert(0, os.path.abspath('../../peakweather')) 

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PeakWeather'
copyright = '2025, Ivan Marisca, Michele Cattaneo, Daniele Zambon'
author = 'Ivan Marisca, Michele Cattaneo, Daniele Zambon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2480A8",
        "color-brand-content": "#2480A8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#2A9BCB",
        "color-brand-content": "#2A9BCB",
    }
}
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = "_static/peakweather_logo.png"
html_favicon = '_static/peakweather_logo.png'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'jupyter_notebook',
}

# pygments_style = "dracula"

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ("https://numpy.org/devdocs/", None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}
