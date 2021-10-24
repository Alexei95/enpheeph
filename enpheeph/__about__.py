# taken as example from PyTorch Lightning

import time

# these two variables will not be imported with import * as they start with _
_this_year = time.strftime("%Y")
_start_year = "2020"

__author__ = "Alessio Colucci"
__author_email__ = "alessio.colucci@protonmail.com"
__copyright__ = f"Copyright (c) {_start_year}-{_this_year}, {__author__}."

# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "Neural Fault Injection Framework"
__docs_url__ = ""

__homepage__ = "https://github.com/Alexei95/enpheeph"

__license__ = "Affero "  # complete this

__long_docs__ = """
TO BE WRITTEN
"""
__long_docs_content_type__ = "text/markdown"
# these urls are used for project_urls in setup
__project_urls__ = {"Changelog": ""}

__version__ = "0.1"

# this must be at the end to import all the previously defined variables when using
# from __about__ import *
# if __all__ is not defined, all the variables will be imported, except the ones
# starting with _, so we need to defined them here
__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
    "__long_docs__",
    "__long_docs_content_type__",
    "__project_urls__",
    "__version__",
]
