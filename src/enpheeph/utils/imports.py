# -*- coding: utf-8 -*-
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

# here we could use importlib.resources but it does not provide the get_distribution
# method, so we keep using pkg_resources for now
# we can use importlib.metadata.distribution, as we only need the version from
# pkg_resources, or even importlib.metadata.version
import importlib.metadata
import importlib.util
import typing

import packaging.requirements
import packaging.specifiers
import packaging.version


# we use the spec from importlib to check the availability of a library
# if it is not None it exists
def is_module_available(module_name: str) -> bool:
    # we check the spec for the presence of a library
    try:
        return importlib.util.find_spec(name=module_name) is not None
    except ModuleNotFoundError:
        return False


# to compare version we use the packaging specifier which checks
# if the found version from the installed package is compatible with the given
# specifier
def compare_version(
    module_name: str,
    version_specifier: packaging.specifiers.SpecifierSet,
) -> bool:
    if not is_module_available(module_name=module_name):
        return False
    version = packaging.version.parse(importlib.metadata.version(module_name))
    return version_specifier.contains(version)


# for checking the availability we simply compare with the requirements
# for extra flags it is as easy as parsing a custom requirements and
# getting the specifier
_enpheeph_raw_requirements = importlib.metadata.requires("enpheeph")
ENPHEEPH_REQUIREMENTS: typing.Tuple[packaging.requirements.Requirement, ...] = tuple(
    packaging.requirements.Requirement(_req)
    for _req in (
        _enpheeph_raw_requirements if _enpheeph_raw_requirements is not None else ()
    )
)

CUPY_NAME: str = "cupy"
NUMPY_NAME: str = "numpy"
NORSE_NAME: str = "norse"
PYTORCH_LIGHTNING_NAME: str = "pytorch_lightning"
SQLALCHEMY_NAME: str = "sqlalchemy"
TORCH_NAME: str = "torch"

# here we have the list of the modules which need to be checked for availability
# custom checks can be done on other values/requirements as well if needed
MODULE_AVAILABILITY_TO_CHECK: typing.Tuple[str, ...] = (
    CUPY_NAME,
    NUMPY_NAME,
    NORSE_NAME,
    PYTORCH_LIGHTNING_NAME,
    SQLALCHEMY_NAME,
    TORCH_NAME,
)
MODULE_AVAILABILITY: typing.Dict[str, bool] = {}
for _mod_name in MODULE_AVAILABILITY_TO_CHECK:
    # we use next on filter as filter is a generator so using next we get the first
    # value, which supposedly should also be the only one
    _version_specifier: packaging.specifiers.SpecifierSet = next(
        filter(lambda x: x.name == _mod_name, ENPHEEPH_REQUIREMENTS)
    ).specifier
    MODULE_AVAILABILITY[_mod_name] = compare_version(
        module_name=_mod_name, version_specifier=_version_specifier
    )
