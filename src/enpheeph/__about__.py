# -*- coding: utf-8 -*-
# taken as example from PyTorch Lightning

# use importlib.metadata to gather the info from the package information
# these are saved in setup.py until we can use pyproject.toml
# import importlib.metadata
import time

# these two variables will not be imported with import * as they start with _
_this_year = time.strftime("%Y")
_start_year = "2020"
