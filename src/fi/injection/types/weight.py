import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype


@src.fi.injection.injectioncallback.InjectionCallback.register_decorator(
        src.fi.utils.enums.parametertype.ParameterType.Weight
)
def init_weight(fault: 'src.fi.injection.faultdescriptor.FaultDescriptor',
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
