# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../mpify/'))
autodoc_mock_imports = ["torch", "multiprocess"]


# -- Project information -----------------------------------------------------

project = 'mpify'
copyright = '2020, Phillip K.S. Chu'
author = 'Phillip K.S. Chu'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': 'https://mpify.readthedocs.io',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

github_url = 'https://github.com/philtrade/mpify'

extlinks = {
    'issue': ('https://github.com/philtrade/mpify/issues/%s', 'issue '),
    'pull': ('https://github.com/philtrade/mpify/pull/%s', 'pull '),
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
"""
autoclass_content = 'both'
autodoc_member_order = "bysource"
autodoc_default_options = {
    'members': True,
    'undoc-members': True
}


def onDocstring(app, what, name, obj, options, lines):
    if not lines:
        return
    if lines[0].startswith('Alias for field number'):
        # strip useless namedtuple number fields
        del lines[:]


def setup(app):
    app.connect('autodoc-process-docstring', onDocstring),
"""
