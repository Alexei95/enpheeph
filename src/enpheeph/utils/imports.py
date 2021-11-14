# -*- coding: utf-8 -*-
import importlib.util
import typing

import pkg_resources


def is_module_available(module_name: str) -> bool:
    # we check the spec for the presence of a library
    try:
        return importlib.util.find_spec(name=module_name) is not None
    except ModuleNotFoundError:
        return False


def compare_version(
    module_name: str,
    version_comparator: typing.Callable[
        [pkg_resources.packaging.version.Version], bool
    ],
) -> bool:
    if not is_module_available(module_name=module_name):
        return False
    version = pkg_resources.parse_version(
        pkg_resources.get_distribution(module_name).version
    )
    return version_comparator(version)


CUPY_MIN_VERSION: str = "9.0.0"
_cupy_version_comparator: typing.Callable[
    [pkg_resources.packaging.version.Version], bool
] = lambda x: x >= pkg_resources.parse_version(CUPY_MIN_VERSION)
IS_CUPY_AVAILABLE: bool = compare_version("cupy", _cupy_version_comparator)

NUMPY_MIN_VERSION: str = "1.19"
_numpy_version_comparator: typing.Callable[
    [pkg_resources.packaging.version.Version], bool
] = lambda x: x >= pkg_resources.parse_version(NUMPY_MIN_VERSION)
IS_NUMPY_AVAILABLE: bool = compare_version("numpy", _numpy_version_comparator)

TORCH_MIN_VERSION: str = "1.8"
_torch_version_comparator: typing.Callable[
    [pkg_resources.packaging.version.Version], bool
] = lambda x: x >= pkg_resources.parse_version(TORCH_MIN_VERSION)
IS_TORCH_AVAILABLE: bool = compare_version("torch", _torch_version_comparator)
