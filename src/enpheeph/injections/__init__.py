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
        "injectionabc",
        "mixins",
        "outputpytorchfault",
        "outputpytorchmonitor",
        "plugins",
        "pytorchinjectionabc",
        "snnoutputnorsefault",
    },
    submod_attrs={
        "injectionabc": [
            "InjectionABC",
        ],
        "mixins": [
            "PyTorchMaskMixin",
            "PyTorchMonitorPostProcessorMixin",
            "pytorchmaskmixin",
            "pytorchmonitorpostprocessormixin",
            "torch_geometric_mean",
        ],
        "outputpytorchfault": [
            "OutputPyTorchFault",
        ],
        "outputpytorchmonitor": [
            "OutputPyTorchMonitor",
        ],
        "plugins": [
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
        ],
        "pytorchinjectionabc": [
            "PyTorchInjectionABC",
        ],
        "snnoutputnorsefault": [
            "SNNOutputNorseFault",
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
    "InjectionABC",
    "InjectionProtocol",
    "LowLevelTorchMaskPluginABC",
    "Monitor",
    "MonitorBaseMixin",
    "MonitorProtocol",
    "NumPyPyTorchMaskPlugin",
    "OutputPyTorchFault",
    "OutputPyTorchMonitor",
    "PolymorphicMixin",
    "PyTorchInjectionABC",
    "PyTorchMaskMixin",
    "PyTorchMonitorPostProcessorMixin",
    "SNNOutputNorseFault",
    "SQLStoragePluginABC",
    "SQLiteStoragePlugin",
    "StoragePluginABC",
    "cupypytorchmaskplugin",
    "fix_pysqlite",
    "injectionabc",
    "lowleveltorchmaskpluginabc",
    "mask",
    "mixins",
    "numpypytorchmaskplugin",
    "outputpytorchfault",
    "outputpytorchmonitor",
    "plugins",
    "pysqlite_begin_emission_fix_on_connect",
    "pytorchinjectionabc",
    "pytorchmaskmixin",
    "pytorchmonitorpostprocessormixin",
    "set_sqlite_pragma",
    "snnoutputnorsefault",
    "sql_data_classes",
    "sqlalchemy_begin_emission_pysqlite",
    "sqlitestorageplugin",
    "sqlstorageplugin",
    "sqlstorageplugineabc",
    "sqlutils",
    "storage",
    "storage_typings",
    "storagepluginabc",
    "torch_geometric_mean",
]
# </AUTOGEN_INIT>
