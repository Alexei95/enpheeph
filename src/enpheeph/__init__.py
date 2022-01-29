# -*- coding: utf-8 -*-
# we ignore mypy/flake8/black as this file is autogenerated
# we ignore this specific error because of AUTOGEN_INIT
# type: ignore[no-untyped-def]
# the following flake8 syntax is wrong, as it will be read as generic noqa, but we use
# it to remember the errors appearing in the __init__.py
# additionally this is not caught by pygrep-hooks as it counts only "type: ignore" and
# "noqa", both with starting #
# flake8: noqa: E302,E305
# fmt: off

# this is required so that the mkinit script will generate the init imports only in this
# section
# <AUTOGEN_INIT>

def lazy_import(module_name, submodules, submod_attrs):
    import importlib
    import os
    name_to_submod = {
        func: mod for mod, funcs in submod_attrs.items()
        for func in funcs
    }

    def __getattr__(name):
        if name in submodules:
            attr = importlib.import_module(
                '{module_name}.{name}'.format(
                    module_name=module_name, name=name)
            )
        elif name in name_to_submod:
            submodname = name_to_submod[name]
            module = importlib.import_module(
                '{module_name}.{submodname}'.format(
                    module_name=module_name, submodname=submodname)
            )
            attr = getattr(module, name)
        else:
            raise AttributeError(
                'No {module_name} attribute {name}'.format(
                    module_name=module_name, name=name))
        globals()[name] = attr
        return attr

    if os.environ.get('EAGER_IMPORT', ''):
        for name in name_to_submod.values():
            __getattr__(name)

        for attrs in submod_attrs.values():
            for attr in attrs:
                __getattr__(attr)
    return __getattr__


__getattr__ = lazy_import(
    __name__,
    submodules={
        'handlers',
        'injections',
        'integrations',
        'utils',
    },
    submod_attrs={
        'handlers': [
            'InjectionHandler',
            'LibraryHandlerPluginABC',
            'PyTorchHandlerPlugin',
            'injectionhandler',
            'libraryhandlerpluginabc',
            'plugins',
            'pytorchhandlerplugin',
        ],
        'injections': [
            'AutoPyTorchMaskPlugin',
            'CuPyPyTorchMaskPlugin',
            'CustomBase',
            'CustomBaseClass',
            'DenseSparseOutputPyTorchFault',
            'ExperimentRun',
            'ExperimentRunBaseMixin',
            'ExperimentRunProtocol',
            'Fault',
            'FaultABC',
            'FaultBaseMixin',
            'FaultProtocol',
            'IndexingPlugin',
            'IndexingPluginABC',
            'Injection',
            'InjectionABC',
            'InjectionProtocol',
            'LowLevelTorchMaskPluginABC',
            'Monitor',
            'MonitorABC',
            'MonitorBaseMixin',
            'MonitorProtocol',
            'NumPyPyTorchMaskPlugin',
            'OutputPyTorchFault',
            'OutputPyTorchMonitor',
            'PolymorphicMixin',
            'PyTorchInjectionABC',
            'PyTorchMaskMixin',
            'PyTorchMonitorPostProcessorMixin',
            'PyTorchSparseInterfaceMixin',
            'PyTorchTensorObjectValidatorMixin',
            'SNNOutputNorseFault',
            'SQLStoragePluginABC',
            'SQLiteStoragePlugin',
            'Session',
            'SessionBaseMixin',
            'StoragePluginABC',
            'WeightPyTorchFault',
            'autopytorchmaskplugin',
            'cupypytorchmaskplugin',
            'densesparseoutputpytorchfault',
            'faultabc',
            'fix_pysqlite',
            'indexing',
            'indexingplugin',
            'indexingpluginabc',
            'injectionabc',
            'lowleveltorchmaskpluginabc',
            'mask',
            'mixins',
            'monitorabc',
            'numpypytorchmaskplugin',
            'outputpytorchfault',
            'outputpytorchmonitor',
            'plugins',
            'pysqlite_begin_emission_fix_on_connect',
            'pytorchinjectionabc',
            'pytorchmaskmixin',
            'pytorchmonitorpostprocessormixin',
            'pytorchsparseinterfacemixin',
            'pytorchtensorobjectvalidatormixin',
            'set_sqlite_pragma',
            'snnoutputnorsefault',
            'sql_data_classes',
            'sqlalchemy_begin_emission_pysqlite',
            'sqlitestorageplugin',
            'sqlstorageplugin',
            'sqlstoragepluginabc',
            'sqlutils',
            'storage',
            'storage_typings',
            'storagepluginabc',
            'torch_geometric_mean',
            'weightpytorchfault',
        ],
        'integrations': [
            'InjectionCallback',
            'injectioncallback',
            'pytorchlightning',
        ],
        'utils': [
            'ActiveDimensionIndexType',
            'AnyIndexType',
            'ArrayType',
            'BaseInjectionLocation',
            'BitFaultMaskInfo',
            'BitFaultValue',
            'BitIndexInfo',
            'BitWidth',
            'DimensionDictType',
            'DimensionIndexType',
            'DimensionLocationIndexType',
            'DimensionType',
            'Endianness',
            'FaultLocation',
            'FaultLocationMixin',
            'FaultMaskOperation',
            'FaultMaskValue',
            'FunctionCallerNameMixin',
            'HandlerStatus',
            'IDGenerator',
            'IDGeneratorSubclass',
            'Index1DType',
            'IndexMultiDType',
            'IndexTimeType',
            'InjectionLocationABC',
            'LocationMixin',
            'LocationModuleNameMixin',
            'LocationOptionalMixin',
            'LowLevelMaskArrayType',
            'ModelType',
            'MonitorLocation',
            'MonitorMetric',
            'ParameterType',
            'PathType',
            'Positive',
            'ShapeType',
            'TensorType',
            'camel_to_snake',
            'classes',
            'compare_version',
            'constants',
            'data_classes',
            'enums',
            'functions',
            'get_object_library',
            'imports',
            'is_module_available',
            'typings',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['ActiveDimensionIndexType', 'AnyIndexType', 'ArrayType',
           'AutoPyTorchMaskPlugin', 'BaseInjectionLocation',
           'BitFaultMaskInfo', 'BitFaultValue', 'BitIndexInfo', 'BitWidth',
           'CuPyPyTorchMaskPlugin', 'CustomBase', 'CustomBaseClass',
           'DenseSparseOutputPyTorchFault', 'DimensionDictType',
           'DimensionIndexType', 'DimensionLocationIndexType', 'DimensionType',
           'Endianness', 'ExperimentRun', 'ExperimentRunBaseMixin',
           'ExperimentRunProtocol', 'Fault', 'FaultABC', 'FaultBaseMixin',
           'FaultLocation', 'FaultLocationMixin', 'FaultMaskOperation',
           'FaultMaskValue', 'FaultProtocol', 'FunctionCallerNameMixin',
           'HandlerStatus', 'IDGenerator', 'IDGeneratorSubclass',
           'Index1DType', 'IndexMultiDType', 'IndexTimeType', 'IndexingPlugin',
           'IndexingPluginABC', 'Injection', 'InjectionABC',
           'InjectionCallback', 'InjectionHandler', 'InjectionLocationABC',
           'InjectionProtocol', 'LibraryHandlerPluginABC', 'LocationMixin',
           'LocationModuleNameMixin', 'LocationOptionalMixin',
           'LowLevelMaskArrayType', 'LowLevelTorchMaskPluginABC', 'ModelType',
           'Monitor', 'MonitorABC', 'MonitorBaseMixin', 'MonitorLocation',
           'MonitorMetric', 'MonitorProtocol', 'NumPyPyTorchMaskPlugin',
           'OutputPyTorchFault', 'OutputPyTorchMonitor', 'ParameterType',
           'PathType', 'PolymorphicMixin', 'Positive', 'PyTorchHandlerPlugin',
           'PyTorchInjectionABC', 'PyTorchMaskMixin',
           'PyTorchMonitorPostProcessorMixin', 'PyTorchSparseInterfaceMixin',
           'PyTorchTensorObjectValidatorMixin', 'SNNOutputNorseFault',
           'SQLStoragePluginABC', 'SQLiteStoragePlugin', 'Session',
           'SessionBaseMixin', 'ShapeType', 'StoragePluginABC', 'TensorType',
           'WeightPyTorchFault', 'autopytorchmaskplugin', 'camel_to_snake',
           'classes', 'compare_version', 'constants', 'cupypytorchmaskplugin',
           'data_classes', 'densesparseoutputpytorchfault', 'enums',
           'faultabc', 'fix_pysqlite', 'functions', 'get_object_library',
           'handlers', 'imports', 'indexing', 'indexingplugin',
           'indexingpluginabc', 'injectionabc', 'injectioncallback',
           'injectionhandler', 'injections', 'integrations',
           'is_module_available', 'libraryhandlerpluginabc',
           'lowleveltorchmaskpluginabc', 'mask', 'mixins', 'monitorabc',
           'numpypytorchmaskplugin', 'outputpytorchfault',
           'outputpytorchmonitor', 'plugins',
           'pysqlite_begin_emission_fix_on_connect', 'pytorchhandlerplugin',
           'pytorchinjectionabc', 'pytorchlightning', 'pytorchmaskmixin',
           'pytorchmonitorpostprocessormixin', 'pytorchsparseinterfacemixin',
           'pytorchtensorobjectvalidatormixin', 'set_sqlite_pragma',
           'snnoutputnorsefault', 'sql_data_classes',
           'sqlalchemy_begin_emission_pysqlite', 'sqlitestorageplugin',
           'sqlstorageplugin', 'sqlstoragepluginabc', 'sqlutils', 'storage',
           'storage_typings', 'storagepluginabc', 'torch_geometric_mean',
           'typings', 'utils', 'weightpytorchfault']
# </AUTOGEN_INIT>
