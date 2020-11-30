import typing

from . import faultmodelabc


class SequentialFaultModel(faultmodelabc.FaultModelABC):
    submodels: typing.List[faultmodelabc.FaultModelABC]

    def __init__(self, submodels, *args, *kwargs):
        kwargs.update(submodels=submodels)

        super().__init__(*args, **kwargs)

        self.submodels = submodels

    def __call__(self, fault_input, *args, **kwargs):
        x = fault_input
        for submodel in self.submodels:
            x = submodel(x)
        return x
