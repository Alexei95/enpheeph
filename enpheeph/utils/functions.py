import functools
import importlib
import types
import typing


def safe_import(library_name: str, package: str = None):
    return importlib.import_module(library_name, package=package)


def test_library_access_wrapper(library: types.ModuleType, library_name: str):
    def decorator(fn: typing.Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if library is None:
                raise RuntimeError(
                        f"{library_name} cannot be imported, "
                        "please check the installation to use this plugin"
                )
            return fn(*args, **kwargs)
        return wrapper
    return decorator