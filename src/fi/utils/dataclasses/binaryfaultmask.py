import dataclasses
import typing

import src.fi.utils.enums.binaryfaultmaskop


@dataclasses.dataclass(init=True, repr=True)
class BinaryFaultMask(object):
    # here we know the default
    DEFAULT_FILL_VALUE = {
        src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.OR: 0,
        src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.AND: 1,
        src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.XOR: 0,
    }

    mask: str
    operation: src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp

    @property
    def fill_value(self):
        return self.DEFAULT_FILL_VALUE[self.operation]
