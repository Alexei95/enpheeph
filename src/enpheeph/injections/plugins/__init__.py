# -*- coding: utf-8 -*-
# we ignore mypy/flake8/black as this file is autogenerated
# type: ignore
# flake: noqa
# fmt: off

# this is required so that the mkinit script will generate the init imports only in this
# section
# <AUTOGEN_INIT>


def lazy_import(module_name, submodules, submod_attrs):
    import importlib
    import os

    name_to_submod = {
        func: mod for mod, funcs in submod_attrs.items() for func in funcs
    }

    def __getattr__(name):
        if name in submodules:
            attr = importlib.import_module(
                "{module_name}.{name}".format(module_name=module_name, name=name)
            )
        elif name in name_to_submod:
            submodname = name_to_submod[name]
            module = importlib.import_module(
                "{module_name}.{submodname}".format(
                    module_name=module_name, submodname=submodname
                )
            )
            attr = getattr(module, name)
        else:
            raise AttributeError(
                "No {module_name} attribute {name}".format(
                    module_name=module_name, name=name
                )
            )
        globals()[name] = attr
        return attr

    if os.environ.get("EAGER_IMPORT", ""):
        for name in name_to_submod.values():
            __getattr__(name)

        for attrs in submod_attrs.values():
            for attr in attrs:
                __getattr__(attr)
    return __getattr__


__getattr__ = lazy_import(
    __name__,
    submodules={
        "mask",
        "storage",
    },
    submod_attrs={
        "mask": [
            "CuPyPyTorchMaskPlugin",
            "LowLevelTorchMaskPluginABC",
            "NumPyPyTorchMaskPlugin",
            "cupypytorchmaskplugin",
            "lowleveltorchmaskpluginabc",
            "numpypytorchmaskplugin",
        ],
        "storage": [
            "CustomBase",
            "CustomBaseClass",
            "ExperimentRun",
            "ExperimentRunBaseMixin",
            "ExperimentRunProtocol",
            "Fault",
            "FaultBaseMixin",
            "FaultProtocol",
            "Injection",
            "InjectionProtocol",
            "Monitor",
            "MonitorBaseMixin",
            "MonitorProtocol",
            "PolymorphicMixin",
            "SQLStoragePluginABC",
            "SQLiteStoragePlugin",
            "StoragePluginABC",
            "fix_pysqlite",
            "pysqlite_begin_emission_fix_on_connect",
            "set_sqlite_pragma",
            "sql_data_classes",
            "sqlalchemy_begin_emission_pysqlite",
            "sqlitestorageplugin",
            "sqlstorageplugin",
            "sqlstorageplugineabc",
            "sqlutils",
            "storage_typings",
            "storagepluginabc",
        ],
    },
)


def __dir__():
    return __all__


__all__ = [
    "CuPyPyTorchMaskPlugin",
    "CustomBase",
    "CustomBaseClass",
    "ExperimentRun",
    "ExperimentRunBaseMixin",
    "ExperimentRunProtocol",
    "Fault",
    "FaultBaseMixin",
    "FaultProtocol",
    "Injection",
    "InjectionProtocol",
    "LowLevelTorchMaskPluginABC",
    "Monitor",
    "MonitorBaseMixin",
    "MonitorProtocol",
    "NumPyPyTorchMaskPlugin",
    "PolymorphicMixin",
    "SQLStoragePluginABC",
    "SQLiteStoragePlugin",
    "StoragePluginABC",
    "cupypytorchmaskplugin",
    "fix_pysqlite",
    "lowleveltorchmaskpluginabc",
    "mask",
    "numpypytorchmaskplugin",
    "pysqlite_begin_emission_fix_on_connect",
    "set_sqlite_pragma",
    "sql_data_classes",
    "sqlalchemy_begin_emission_pysqlite",
    "sqlitestorageplugin",
    "sqlstorageplugin",
    "sqlstorageplugineabc",
    "sqlutils",
    "storage",
    "storage_typings",
    "storagepluginabc",
]
# </AUTOGEN_INIT>
