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
        'helpers',
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
        'helpers': [
            'CaptumSensitivityAnalysis',
            'FaultModelABC',
            'ImportanceSampling',
            'LayerSummaryTorchinfo',
            'ModelSummaryABC',
            'ModelSummaryTorchinfo',
            'abc',
            'captumsensitivityanalysis',
            'faultmodel',
            'faultmodelabc',
            'faultmodels',
            'layersummaryabc',
            'layersummarytorchinfo',
            'modelsummaryabc',
            'modelsummarytorchinfo',
            'plugins',
            'sensitivityanalysis',
            'summaries',
        ],
        'injections': [
            'AutoPyTorchMaskPlugin',
            'CSVStoragePluginABC',
            'CuPyPyTorchMaskPlugin',
            'CustomBase',
            'CustomBaseClass',
            'DenseSparseOutputPyTorchFault',
            'ExperimentRun',
            'ExperimentRunBaseMixin',
            'ExperimentRunProtocol',
            'FPQuantizedOutputPyTorchFault',
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
            'PandasCSVStoragePlugin',
            'PolymorphicMixin',
            'PrunedDenseToSparseWeightPyTorchFault',
            'PyTorchInjectionABC',
            'PyTorchMaskMixin',
            'PyTorchMonitorPostProcessorMixin',
            'PyTorchSparseInterfaceMixin',
            'PyTorchSparseInterfacePluginABC',
            'PyTorchTensorObjectValidatorMixin',
            'QuantizedOutputPyTorchFault',
            'SNNOutputNorseFault',
            'SQLStoragePluginABC',
            'SQLiteStoragePlugin',
            'Session',
            'SessionBaseMixin',
            'SessionProtocol',
            'StoragePluginABC',
            'WeightPyTorchFault',
            'abc',
            'autopytorchmaskplugin',
            'csv',
            'csvdataclasses',
            'csvstorageplugin',
            'csvstoragepluginabc',
            'cupypytorchmaskplugin',
            'densesparseoutputpytorchfault',
            'faultabc',
            'fix_pysqlite',
            'fpquantizedoutputpytorchfault',
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
            'pruneddensetosparseactivationpytorchfault',
            'pruneddensetosparseweightpytorchfault',
            'pysqlite_begin_emission_fix_on_connect',
            'pytorchinjectionabc',
            'pytorchmaskmixin',
            'pytorchmonitorpostprocessormixin',
            'pytorchquantizationmixin',
            'pytorchsparseinterfacemixin',
            'pytorchsparseinterfacepluginabc',
            'pytorchtensorobjectvalidatormixin',
            'quantizedoutputpytorchfault',
            'set_sqlite_pragma',
            'snnoutputnorsefault',
            'sparse',
            'sql',
            'sqlalchemy_begin_emission_pysqlite',
            'sqldataclasses',
            'sqlitestorageplugin',
            'sqlstoragepluginabc',
            'sqlutils',
            'storage',
            'storagepluginabc',
            'storagetypings',
            'torch_geometric_mean',
            'utils',
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
            'AnyMaskType',
            'ArrayType',
            'BaseInjectionLocation',
            'BitFaultMaskInfo',
            'BitFaultValue',
            'BitIndexInfo',
            'BitWidth',
            'DimensionDictType',
            'DimensionIndexType',
            'DimensionLocationIndexType',
            'DimensionLocationMaskType',
            'DimensionType',
            'Endianness',
            'FaultLocation',
            'FaultLocationMixin',
            'FaultMaskOperation',
            'FaultMaskValue',
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
            'Mask1DType',
            'MaskMultiDType',
            'ModelType',
            'MonitorLocation',
            'MonitorMetric',
            'ParameterType',
            'PathType',
            'ShapeType',
            'SkipIfErrorContextManager',
            'TensorType',
            'camel_to_snake',
            'classes',
            'compare_version',
            'constants',
            'dataclasses',
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

__all__ = ['ActiveDimensionIndexType', 'AnyIndexType', 'AnyMaskType',
           'ArrayType', 'AutoPyTorchMaskPlugin', 'BaseInjectionLocation',
           'BitFaultMaskInfo', 'BitFaultValue', 'BitIndexInfo', 'BitWidth',
           'CSVStoragePluginABC', 'CaptumSensitivityAnalysis',
           'CuPyPyTorchMaskPlugin', 'CustomBase', 'CustomBaseClass',
           'DenseSparseOutputPyTorchFault', 'DimensionDictType',
           'DimensionIndexType', 'DimensionLocationIndexType',
           'DimensionLocationMaskType', 'DimensionType', 'Endianness',
           'ExperimentRun', 'ExperimentRunBaseMixin', 'ExperimentRunProtocol',
           'FPQuantizedOutputPyTorchFault', 'Fault', 'FaultABC',
           'FaultBaseMixin', 'FaultLocation', 'FaultLocationMixin',
           'FaultMaskOperation', 'FaultMaskValue', 'FaultModelABC',
           'FaultProtocol', 'HandlerStatus', 'IDGenerator',
           'IDGeneratorSubclass', 'ImportanceSampling', 'Index1DType',
           'IndexMultiDType', 'IndexTimeType', 'IndexingPlugin',
           'IndexingPluginABC', 'Injection', 'InjectionABC',
           'InjectionCallback', 'InjectionHandler', 'InjectionLocationABC',
           'InjectionProtocol', 'LayerSummaryTorchinfo',
           'LibraryHandlerPluginABC', 'LocationMixin',
           'LocationModuleNameMixin', 'LocationOptionalMixin',
           'LowLevelMaskArrayType', 'LowLevelTorchMaskPluginABC', 'Mask1DType',
           'MaskMultiDType', 'ModelSummaryABC', 'ModelSummaryTorchinfo',
           'ModelType', 'Monitor', 'MonitorABC', 'MonitorBaseMixin',
           'MonitorLocation', 'MonitorMetric', 'MonitorProtocol',
           'NumPyPyTorchMaskPlugin', 'OutputPyTorchFault',
           'OutputPyTorchMonitor', 'PandasCSVStoragePlugin', 'ParameterType',
           'PathType', 'PolymorphicMixin',
           'PrunedDenseToSparseWeightPyTorchFault', 'PyTorchHandlerPlugin',
           'PyTorchInjectionABC', 'PyTorchMaskMixin',
           'PyTorchMonitorPostProcessorMixin', 'PyTorchSparseInterfaceMixin',
           'PyTorchSparseInterfacePluginABC',
           'PyTorchTensorObjectValidatorMixin', 'QuantizedOutputPyTorchFault',
           'SNNOutputNorseFault', 'SQLStoragePluginABC', 'SQLiteStoragePlugin',
           'Session', 'SessionBaseMixin', 'SessionProtocol', 'ShapeType',
           'SkipIfErrorContextManager', 'StoragePluginABC', 'TensorType',
           'WeightPyTorchFault', 'abc', 'autopytorchmaskplugin',
           'camel_to_snake', 'captumsensitivityanalysis', 'classes',
           'compare_version', 'constants', 'csv', 'csvdataclasses',
           'csvstorageplugin', 'csvstoragepluginabc', 'cupypytorchmaskplugin',
           'dataclasses', 'densesparseoutputpytorchfault', 'enums', 'faultabc',
           'faultmodel', 'faultmodelabc', 'faultmodels', 'fix_pysqlite',
           'fpquantizedoutputpytorchfault', 'functions', 'get_object_library',
           'handlers', 'helpers', 'imports', 'indexing', 'indexingplugin',
           'indexingpluginabc', 'injectionabc', 'injectioncallback',
           'injectionhandler', 'injections', 'integrations',
           'is_module_available', 'layersummaryabc', 'layersummarytorchinfo',
           'libraryhandlerpluginabc', 'lowleveltorchmaskpluginabc', 'mask',
           'mixins', 'modelsummaryabc', 'modelsummarytorchinfo', 'monitorabc',
           'numpypytorchmaskplugin', 'outputpytorchfault',
           'outputpytorchmonitor', 'plugins',
           'pruneddensetosparseactivationpytorchfault',
           'pruneddensetosparseweightpytorchfault',
           'pysqlite_begin_emission_fix_on_connect', 'pytorchhandlerplugin',
           'pytorchinjectionabc', 'pytorchlightning', 'pytorchmaskmixin',
           'pytorchmonitorpostprocessormixin', 'pytorchquantizationmixin',
           'pytorchsparseinterfacemixin', 'pytorchsparseinterfacepluginabc',
           'pytorchtensorobjectvalidatormixin', 'quantizedoutputpytorchfault',
           'sensitivityanalysis', 'set_sqlite_pragma', 'snnoutputnorsefault',
           'sparse', 'sql', 'sqlalchemy_begin_emission_pysqlite',
           'sqldataclasses', 'sqlitestorageplugin', 'sqlstoragepluginabc',
           'sqlutils', 'storage', 'storagepluginabc', 'storagetypings',
           'summaries', 'torch_geometric_mean', 'typings', 'utils',
           'weightpytorchfault']
# </AUTOGEN_INIT>
