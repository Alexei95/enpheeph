import collections
import copy
import dataclasses
import typing

import pytorch_lightning
import torch

import src.dispatcherabc
import src.fi.faultdescriptor
import src.fi.fiutils
import src.fi.mixins.moduleupdater


# we integrate both the basic injection callback together with the dispatcher
@dataclasses.dataclass(init=True, repr=True)
class InjectionCallback(
        pytorch_lightning.Callback,
        src.mixins.dispatcher.Dispatcher,
        src.fi.mixins.moduleupdater.ModuleUpdater
):

    # list of all faults to be injected, do not use two active faults on the
    # same module
    fault_descriptor_list: typing.Sequence[
            'src.fi.faultdescriptor.FaultDescriptor'
    ] = dataclasses.field(init=True, repr=True, default_factory=[])
    # enable/disable the injection, can be changed with the functions
    enabled: bool = dataclasses.field(init=True, repr=True, default=True)
    # enable the automatic init of the top module during each on_test_start
    auto_model_init_on_test_start: bool = dataclasses.field(
            init=True, repr=True, default=True
    )
    # enabled faults, if None at init then all the faults are enabled, and they
    # follow the general flag for the callback
    # otherwise it must be a valid sequence of faults
    enabled_faults: typing.Union[
            type(Ellipsis),
            typing.Sequence[
                    'src.fi.faultdescriptor.FaultDescriptor'
            ],
     ] = dataclasses.field(init=True, repr=False, default=...)
    # internal flag for checking whether we have set up the modules for
    # injection
    _active: bool = dataclasses.field(init=False, repr=False, default=False)
    # dict of injected modules to be loaded in the main model
    _modules: typing.Dict[
            'src.fi.faultdescriptor.FaultDescriptor',
            'torch.nn.Module',
    ] = dataclasses.field(
            init=False,
            repr=False,
            default_factory=dict)
    # dict containing the original modules, before substituting them for
    # the injection
    _modules_backup: typing.Dict[
            'src.fi.faultdescriptor.FaultDescriptor',
            'torch.nn.Module',
    ] = dataclasses.field(
            init=False,
            repr=False,
            default_factory=dict)

    def __post_init__(self):
        # we check all the faults are mapped
        dispatching_dict = self.get_dispatching_dict()
        if not all(
                fault.parameter_type in dispatching_dict
                for fault in self.fault_descriptor_list
        ):
            # FIXME: check error to be raised
            raise ValueError(
                    'Please map all the faults to a fault initializer')

        # if we have an ... then all the faults are enabled
        if isinstance(self.enabled_faults, type(Ellipsis)):
            self.enabled_faults = self.fault_descriptor_list
        # otherwise we have a list and we check that all the enabled faults
        # are actually present in the fault list, otherwise we raise an error
        elif isinstance(self.enabled_faults, collections.abc.Iterable):
            if not all(
                    fault in self.fault_descriptor_list
                    for fault in self.enabled_faults
            ):
                # FIXME: check error to be raised
                raise ValueError(
                        'At least 1 enabled fault is not in the fault list')
            # we convert it to a list
            self.enabled_faults = list(self.enabled_faults)
        # if it is not an ordered container we raise an error
        else:
            # FIXME: check error to be raised
            raise TypeError(
                    'Wrong type for enabled faults, '
                    'must be an ordered container'
            )

    # this function activates the injection, by substituting all the modules
    # with the custom one, following the list
    def on_test_start(self, trainer, pl_module):
        # if we are enabled for injecting and nothing else is running already
        if self.enabled and not self._active:
            # if we are set for auto-init, then we set up the model
            if self.auto_model_init_on_test_start:
                self.init_model(pl_module)
            # we use the ModuleUpdater static method to load all the fault
            # injection modules in the current pl_module
            self.update_module_from_module_list(
                pl_module,
                # we convert the dict with fault: module to module_name: module
                {
                        fault.module_name: module
                        for fault, module in self._modules.items()
                },
                in_place=True)

            # we change the flag after cycling through all the modules
            self._active = True

    # this function deactivates the injection, by substituting all the modules
    # with the backups
    def on_test_end(self, trainer, pl_module):
        # if we are active
        if self._active:
            # we use the ModuleUpdater static method to reload all the backup
            # modules in the current pl_module
            self.update_module_from_module_list(
                pl_module,
                # we convert the dict with fault: module to module_name: module
                {
                        fault.module_name: module
                        for fault, module in self._modules_backup.items()
                },
                in_place=True)

            # we change the flag after cycling through all the modules
            self._active = False

    # this method goes over all the structure of the top module, and calls
    # a function to initialize the injection for each target sub-module
    def init_model(self, top_module: 'torch.nn.Module'):
        # for now we iterate through all of them, but we could build a
        # recursive list to avoid going through a double for loop
        # we consider only enabled faults
        for fault in self.enabled_faults:
            # we use the get_module function from the ModuleUpdater class
            module = self.get_module(fault.module_name, top_module)
            self.init_module(fault, module)

    # this method must be implemented based on the injection type
    # (weight, activations, ...)
    def init_module(
            self,
            fault: 'src.fi.faultdescriptor.FaultDescriptor',
            module: 'torch.nn.Module'
            ):
        # we copy the original module
        self._modules_backup[fault] = copy.deepcopy(module)

        injection_module = self.dispatch_call(
                # name for dispatch_call
                name=fault.parameter_type,
                # parameters for the dispatched call
                fault=fault,
                module=module,
        )

        # we set up the new module to be loaded later on
        self._modules[fault] = copy.deepcopy(injection_module)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    # by default all faults are enabled, here we can pass a list of faults
    # to be enabled, using ... as joker for covering all the faults in the
    # current list
    # it returns the enabled faults
    def enable_faults(self, fault_list: typing.Union[
            type(Ellipsis),
            typing.Sequence[
                    'src.fi.faultdescriptor.FaultDescriptor'
            ],
    ]) -> typing.List['src.fi.faultdescriptor.FaultDescriptor']:
        # if we get an ... we enable all the faults
        if isinstance(fault_list, type(Ellipsis)):
            self.enabled_faults = self.fault_descriptor_list
        elif isinstance(fault_list, collections.abc.Iterable):
            if not all(
                    fault in self.fault_descriptor_list
                    for fault in self.enabled_faults
            ):
                # FIXME: check error to be raised
                raise ValueError(
                        'At least 1 enabled fault is not in the fault list')
            # if we pass the check then we append the list at the end of the
            # currently enabled ones, removing any duplicates
            for fault in fault_list:
                if fault not in self.enabled_faults:
                    self.enabled_faults.append(fault)
        # if it is not an ordered container we raise an error
        else:
            # FIXME: check error to be raised
            raise TypeError(
                    'Wrong type for enabled faults, '
                    'must be an ordered container'
            )

        return self.enabled_faults

    # by default all faults are enabled, here we can pass a list of faults
    # to be disabled, using ... as joker for covering all the faults in the
    # current list
    # it returns the enabled faults
    def disable_faults(self, fault_list: typing.Union[
            type(Ellipsis),
            typing.Sequence[
                    'src.fi.faultdescriptor.FaultDescriptor'
            ],
    ]) -> typing.List['src.fi.faultdescriptor.FaultDescriptor']:
        # if we get an ... we enable all the faults
        if isinstance(fault_list, type(Ellipsis)):
            self.enabled_faults = []
        elif isinstance(fault_list, collections.abc.Iterable):
            if not all(
                    fault in self.fault_descriptor_list
                    for fault in self.enabled_faults
            ):
                # FIXME: check error to be raised
                raise ValueError(
                        'At least 1 enabled fault is not in the fault list')
            # if we pass the check then we remove the elements from the enabled
            # list
            for fault in fault_list:
                self.enabled_faults.remove(fault)
        # if it is not an ordered container we raise an error
        else:
            # FIXME: check error to be raised
            raise TypeError(
                    'Wrong type for enabled faults, '
                    'must be an ordered container'
            )

        return self.enabled_faults



@InjectionCallback.register_decorator(
        src.fi.faultdescriptor.ParameterType.Weight
)
def init_weight(fault: 'src.fi.faultdescriptor.FaultDescriptor',
                module: 'torch.nn.Module') -> 'torch.nn.Module':
    # we get the weights
    weights = getattr(module, fault.parameter_name)
    weights = src.fi.fiutils.inject_tensor_fault_pytorch(
            tensor=weights,
            fault=fault,
    )
    # we set the weights to the updated value
    setattr(module, fault.parameter_name, weights)
    # we return the new module
    return module


# we map the class to the activation injection
@InjectionCallback.register_decorator(
        src.fi.faultdescriptor.ParameterType.Activation
)
# NOTE: we can only have one of the following module per layer, as the parsing
# of the top module is done statically on the original structure, not on the
# updated layers
# NOTE: we cannot have dataclasses go with torch.nn.Module as Module.__init__
# must be called before the dataclasses init
# FIXME: implement also backward for fault-aware training
class ActivationInjectionModule(torch.nn.Module):
    def __init__(
                self,
                fault: 'src.fi.faultdescriptor.FaultDescriptor',
                module: 'torch.nn.Module'

    ):
        super().__init__()

        self.fault = fault
        self.module = module

    def forward(self, x):
        # we get the exact result from the previous module
        y_temp = self.module(x)
        # we inject the faults in the tensor
        y = src.fi.fiutils.inject_tensor_fault_pytorch(
                tensor=y_temp,
                fault=self.fault,
        )
        # we return the fault-injected tensor
        return y
