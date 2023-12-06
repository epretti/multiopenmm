import inspect
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(inspect.getsourcefile(inspect.currentframe()))), "..")))

project = "OpenMMREMD"
author = "Evan Pretti"
version = "0.1.0"
copyright = "2023 The Regents of the University of California"

templates_path = ["_templates"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "classic"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://www.numpy.org/doc/stable", None),
    "openmm": ("http://docs.openmm.org/latest/api-python", None),
}
