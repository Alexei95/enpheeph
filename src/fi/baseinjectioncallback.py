import copy
import dataclasses
import typing

import pytorch_lightning
import torch

from . import basefaultdescriptor
from . import fiutils


def init_weight(fault: basefaultdescriptor.BaseFaultDescriptor,
                module: torch.nn.Module) -> torch.nn.Module:
    # we get the weights
    weights = getattr(module, fault.parameter_name)
    # we get the tensor value to be injected
    original_tensor = weights[fault.tensor_index]
    # to inject the values, we need to flatten the tensor
    flattened_tensor = original_tensor.flatten()
    # then we need to process them one by one
    flattened_bit_tensor = []
    for element in flattened_tensor:
        flattened_bit_tensor.append(fiutils.pytorch_element_to_binary(element))
    # then we inject the faults
    injected_flattened_bit_tensor = []
    for binary in flattened_bit_tensor:
        injected_flattened_bit_tensor.append(fiutils.inject_fault(binary, fault))
    # we convert back to the original data
    injected_flattened_tensor_list = []
    for injected_binary, original_binary in zip(injected_flattened_bit_tensor, flattened_bit_tensor):
        injected_flattened_tensor_list.append(fiutils.binary_to_pytorch_element(injected_binary, original_binary))
    injected_flattened_tensor = torch.Tensor(injected_flattened_tensor_list).to(flattened_tensor)









def init_activation(fault: basefaultdescriptor.BaseFaultDescriptor,
                    module: torch.nn.Module) -> torch.nn.Module:
    pass


# this can be improved with a getter class, implementing all of these and then
# providing an interface for checking the association
PARAMETER_TYPE_MAPPING = {basefaultdescriptor.ParameterType.Weight: init_weight,
                          basefaultdescriptor.ParameterType.Activation: init_activation,}


@dataclasses.dataclass(init=True)
class BaseInjectionCallback(pytorch_lightning.Callback):
    fault_descriptor_list: list[basefaultdescriptor.BaseFaultDescriptor] = []
    enabled: bool = True
    _active: bool = dataclasses.field(init=False, default=False)
    _modules: dict[str, torch.nn.Module] = dataclasses.field(init=False, default=[])
    _modules_backup: dict[str, torch.nn.Module] = dataclasses.field(init=False, default=[])

    def __post_init__(self):
        # we check all the faults are mapped
        for fault in self.fault_descriptor_list:
            assert PARAMETER_TYPE_MAPPING.get(fault.parameter_type, None) is not None

    # this function activates the injection, by substituting all the modules
    # with the custom one, following the list
    def on_test_start(self, trainer, pl_module):
        # if we are enabled for injecting and nothing else is running already
        if self.enabled and not self._active:
            # we need to cycle through all the modules
            # all of them are already backed up, so we simply have to
            # substitute them
            for module_name, module in self._modules.items():
                # we reach the final module whose attribute must be updated
                # by going through the tree until the last-but-one attribute
                dest_module = pl_module
                for submodule in module_name.split('.')[:-1]:
                    dest_module = getattr(dest_module, submodule)

                # the last attribute is used to set the module to the injection
                # one
                setattr(dest_module, module_name.split('.')[-1],
                        module)

            # we change the flag after cycling through all the modules
            self._active = True

    # this function deactivates the injection, by substituting all the modules
    # with the backups
    def on_test_stop(self, trainer, pl_module):
        # if we are active or there is a backup module
        if self._active:
            # we need to cycle through all the backup modules
            for module_name, module in self._modules_backup.items():
                # we reach the final module whose attribute must be updated
                # by going through the tree until the last-but-one attribute
                dest_module = pl_module
                for submodule in module_name.split('.')[:-1]:
                    dest_module = getattr(dest_module, submodule)

                # we reset the module to the non-injected one
                setattr(dest_module, module_name.split('.')[-1],
                        module)

            # we change the flag after cycling through all the modules
            self._active = False

    # this method goes over all the structure of the top module, and calls
    # a function to initialize the injection for each target sub-module
    def init_top_module(self, top_module: torch.nn.Module):
        # for now we iterate through all of them, but we could build a
        # recursive list to avoid going through a double for loop
        for fault in self.fault_descriptor_list:
            module = top_module
            for submodule in fault.module_name.split('.'):
                module = getattr(module, submodule)
            self.init_module(fault, module)

    # this method must be implemented based on the injection type
    # (weight, activations, ...)
    def init_module(self, fault: basefaultdescriptor.BaseFaultDescriptor,
                    module: torch.nn.Module):
        # we copy the original module
        self._modules_backup[fault.module_name] = copy.deepcopy(module)

        injection_module = PARAMETER_TYPE_MAPPING[fault.parameter_type]

        self._modules[fault.module_name] = injection_module

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
