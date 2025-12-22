# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2025 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# for defaultdict
import collections

# use importlib.metadata to gather the info from the package information
# these are saved in setup.py until we can use pyproject.toml
import importlib.metadata
import time

# these two variables will not be imported with import * as they start with _
_this_year = time.strftime("%Y")
_start_year = "2020"

# version, to be accessed by setuptools
__version__ = "0.1.0a1"

# metadata taken from the package through importlib.metadata.metadata
try:
    _metadata = importlib.metadata.metadata("enpheeph")
    _get_all_metadata = _metadata.get_all
except importlib.metadata.PackageNotFoundError:
    # using defaultdict with None, we can use the same object to return
    # the None value for all the attributes
    _metadata = collections.defaultdict(lambda: None)

    def _get_all_metadata(x):
        return []


# here are all the values of the attributes from the package info
name = _metadata["Name"]
version = _metadata["Version"]
classifiers = _get_all_metadata("Classifier")
authors = _get_all_metadata("Author")
author_emails = _get_all_metadata("Author-email")
