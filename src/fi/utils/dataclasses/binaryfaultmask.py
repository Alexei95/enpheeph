import dataclasses
import typing

import src.fi.utils.enums.binaryfaultmaskop


@dataclasses.dataclass(init=True, repr=True)
class BinaryFaultMask(object):
    mask: str
    operation: src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp
