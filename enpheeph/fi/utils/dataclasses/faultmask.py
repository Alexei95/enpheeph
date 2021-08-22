import dataclasses
import typing

import enpheeph.fi.utils.enums.faultmaskop


@dataclasses.dataclass(init=True, repr=True)
class FaultMask(object):
    # here we know the default
    DEFAULT_FILL_VALUE = {
        enpheeph.fi.utils.enums.faultmaskop.FaultMaskOp.OR: 0,
        enpheeph.fi.utils.enums.faultmaskop.FaultMaskOp.AND: 1,
        enpheeph.fi.utils.enums.faultmaskop.FaultMaskOp.XOR: 0,
    }

    # binary is for BinaryHandler
    # numpy/cupy is for NumpyLikeHandler
    # torch is for PyTorchHandler
    mask: typing.Union[str, 'numpy.ndarray', 'cupy.ndarray', 'torch.Tensor']
    operation: enpheeph.fi.utils.enums.faultmaskop.FaultMaskOp

    @property
    def fill_value(self):
        return self.DEFAULT_FILL_VALUE[self.operation]
