"""Configuration file for the Sphinx documentation builder."""

import inspect
import os
import subprocess
import sys

import gradient_accumulator


# Project information
url = "https://github.com/EIDOSLAB/torchstain"

# General configuration
master_doc = 'index'

# Sphinx extension modules
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['build']

# Options for the intersphinx extension
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Draw graphs in the SVG format instead of the default PNG format
graphviz_output_format = 'svg'

# Generate autosummary
autosummary_generate = True

# sphinx.ext.linkcode: Try to link to source code on GitHub
REVISION_CMD = ['git', 'rev-parse', '--short', 'HEAD']
try:
    _git_revision = subprocess.check_output(REVISION_CMD).strip()
except (subprocess.CalledProcessError, OSError):
    _git_revision = 'master'
else:
    _git_revision = _git_revision.decode('utf-8')


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    module = info.get('module', None)
    fullname = info.get('fullname', None)
    if not module or not fullname:
        return None
    obj = sys.modules.get(module, None)
    if obj is None:
        return None

    for part in fullname.split('.'):
        obj = getattr(obj, part)
        if isinstance(obj, property):
            obj = obj.fget
        if hasattr(obj, '__wrapped__'):
            obj = obj.__wrapped__

    file = inspect.getsourcefile(obj)
    package_dir = os.path.dirname(gradient_accumulator.__file__)
    if file is None or os.path.commonpath([file, package_dir]) != package_dir:
        return None
    file = os.path.relpath(file, start=package_dir)
    source, line_start = inspect.getsourcelines(obj)
    line_end = line_start + len(source) - 1
    filename = f'src/gradient_accumulator/{file}#L{line_start}-L{line_end}'
    return f'{url}/blob/{_git_revision}/{filename}'
