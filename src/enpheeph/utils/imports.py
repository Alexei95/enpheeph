# -*- coding: utf-8 -*-
import importlib.metadata
import importlib.util
import typing

import packaging.requirements
import packaging.specifiers
import packaging.version
import pkg_resources


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
    version = packaging.version.parse(
        pkg_resources.get_distribution(module_name).version
    )
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
TORCH_NAME: str = "torch"

# here we have the list of the modules which need to be checked for availability
# custom checks can be done on other values/requirements as well if needed
MODULE_AVAILABILITY_TO_CHECK: typing.Tuple[str, ...] = (
    CUPY_NAME,
    NUMPY_NAME,
    NORSE_NAME,
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
