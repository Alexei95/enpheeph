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
        'pytorchmaskmixin',
        'pytorchmonitorpostprocessormixin',
    },
    submod_attrs={
        'pytorchmaskmixin': [
            'PyTorchMaskMixin',
        ],
        'pytorchmonitorpostprocessormixin': [
            'PyTorchMonitorPostProcessorMixin',
            'torch_geometric_mean',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['PyTorchMaskMixin', 'PyTorchMonitorPostProcessorMixin',
           'pytorchmaskmixin', 'pytorchmonitorpostprocessormixin',
           'torch_geometric_mean']
# </AUTOGEN_INIT>
