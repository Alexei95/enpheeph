import copy
import importlib
import os
import pathlib
import typing


class ModuleGatherer(object):
    MODULE_GLOB = '*.py'
    NOT_MODULE_FILTER = ['__init__.py']

    # this function gather objects from different files in the same directory
    # these objects are squashed together with the update_function, starting
    # from a copy of the default_obj
    @staticmethod
    def gather_objects(
            *,
            path,
            filter_,
            package_name,
            obj_name,
            default_obj,
            update_function,
            glob,
    ):
        res = copy.deepcopy(default_obj)
        # we update the result to contain all the name-class associations in
        # the package
        # NOTE: depending on the glob pattern we can allow recursive
        # exploration
        for m in pathlib.Path(path).glob(glob):
            # if the file is __init__.py or a directory we skip
            if m.name in filter_ or m.is_dir():
                continue
            # we get the full name removing the suffix
            # BUG: without relativeness consideration, if we have
            # directories we cannot import them as we consider only the latest
            # script name
            # FIX: use relative_to(path) and '.'.join(parts) to get the total
            # name for the module, which must be still reachable via __init__
            # imports
            module_name = '.'.join(m.relative_to(path).with_suffix('').parts)
            # we append the package of __init__ for the import
            # FIX: without this package we would be unable to reach the module
            # using relative imports or relative imports inside the module
            # would fail because the module would not know its parent package
            complete_module_name = package_name + '.' + module_name
            # we must also pass the package when importing
            module = importlib.import_module(
                    complete_module_name,
                    package=package_name
            )
            res = update_function(res, getattr(module, obj_name, default_obj))
            del module
        return res

    @classmethod
    def import_submodules(
            cls,
            package_name: str,
            package_path: typing.Union[os.PathLike, str],
            root: typing.Union[os.PathLike, str],
            module_glob: str = None,
            not_module_filter: typing.Sequence[str] = None
    ) -> typing.Dict[str, type(typing)]:
        if module_glob is None:
            module_glob = cls.MODULE_GLOB
        if not_module_filter is None:
            not_module_filter = cls.NOT_MODULE_FILTER

        modules = {}
        # NOTE: depending on the glob pattern we can allow recursive
        # exploration
        for m in (pathlib.Path(root) / package_path).glob(module_glob):
            # if the file is __init__.py or a directory we skip
            if m.name in not_module_filter or m.is_dir():
                continue
            # we get the full name removing the suffix
            # BUG: without relativeness consideration, if we have
            # directories we cannot import them as we consider only the latest
            # script name
            # FIX: use relative_to(root) and '.'.join(parts) to get the total
            # name for the module, which must be still reachable via __init__
            # imports
            module_name = '.'.join(m.relative_to(root).with_suffix('').parts)
            # we compute the package name by using all the elements of the
            # module name except for the last name
            package_name = '.'.join(module_name.split('.')[:-1])
            # we must also pass the package when importing
            # since this module contains all the calls we need, we simply
            # import it
            module = importlib.import_module(
                    module_name,
                    package=package_name
            )
            # we add it to the dict
            modules[module_name] = module

        return modules
