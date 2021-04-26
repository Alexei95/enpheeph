import copy
import dataclasses
import typing

import pytorch_lightning
import torch

import src.dispatcherabc
import src.fi.basefaultdescriptor
import src.fi.fiutils
import src.fi.moduleupdater


# we integrate both the basic injection callback together with the dispatcher
@dataclasses.dataclass(init=True)
class BaseInjectionCallback(
        pytorch_lightning.Callback,
        src.dispatcherabc.DispatcherABC,
        src.fi.moduleupdater.ModuleUpdater):

    # list of all faults to be injected
    fault_descriptor_list: typing.Sequence[
            src.fi.basefaultdescriptor.BaseFaultDescriptor
            ] = dataclasses.field(default_factory=[])
    # enable/disable the injection, can be changed with the functions
    enabled: bool = True
    # enable the automatic init of the top module during each on_test_start
    auto_model_init_on_test_start: bool = True
    # internal flag for checking whether we have set up the modules for
    # injection
    _active: bool = dataclasses.field(init=False, default=False)
    # dict of injected modules to be loaded in the main model
    _modules: typing.Dict[str, torch.nn.Module] = dataclasses.field(
            init=False,
            default_factory=dict)
    # dict containing the original modules, before substituting them for
    # the injection
    _modules_backup: typing.Dict[str, torch.nn.Module] = dataclasses.field(
            init=False,
            default_factory=dict)

    def __post_init__(self):
        # we check all the faults are mapped
        dispatching_dict = self.get_dispatching_dict()
        assert all(fault.parameter_type in dispatching_dict
                   for fault in self.fault_descriptor_list), (
                    'Please map all the faults to a fault initializer')

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
                self._modules,
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
                self._modules_backup,
                in_place=True)

            # we change the flag after cycling through all the modules
            self._active = False

    # this method goes over all the structure of the top module, and calls
    # a function to initialize the injection for each target sub-module
    def init_model(self, top_module: torch.nn.Module):
        # for now we iterate through all of them, but we could build a
        # recursive list to avoid going through a double for loop
        for fault in self.fault_descriptor_list:
            # we use the get_module function from the ModuleUpdater class
            module = self.get_module(fault.module_name, top_module)
            self.init_module(fault, module)

    # this method must be implemented based on the injection type
    # (weight, activations, ...)
    def init_module(
            self,
            fault: src.fi.basefaultdescriptor.BaseFaultDescriptor,
            module: torch.nn.Module
            ):
        # we copy the original module
        self._modules_backup[fault.module_name] = copy.deepcopy(module)

        injection_module = self.dispatch_call(
                # name for dispatch_call
                name=fault.parameter_type,
                # parameters for the dispatched call
                fault=fault,
                module=module,
        )

        # we set up the new module to be loaded later on
        self._modules[fault.module_name] = copy.deepcopy(injection_module)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


@BaseInjectionCallback.register_decorator(
        src.fi.basefaultdescriptor.ParameterType.Weight
)
def init_weight(fault: src.fi.basefaultdescriptor.BaseFaultDescriptor,
                module: torch.nn.Module) -> torch.nn.Module:
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


# NOTE: we can only have one of the following module per layer, as the parsing
# of the top module is done statically on the original structure, not on the
# updated layers
# FIXME: implement also backward for fault-aware training
@dataclasses.dataclass(init=True, repr=True)
class ActivationInjectionModule(torch.nn.Module):
    fault: src.fi.basefaultdescriptor.BaseFaultDescriptor
    module: torch.nn.Module

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
