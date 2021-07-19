import torch

import src.ml.pl.datasets.utils.transforms.transformabc


class ToDtype(src.ml.pl.datasets.utils.transforms.transformabc.TransformABC):
    def __init__(self, dtype):
        super().__init__()

        # this is a workaround to YAML-serialize the dtype
        # the dtype itself cannot be serialized in YAML, and it is required
        # as the transform may be a hparam to be serialized
        self._dtype_holder = torch.zeros(0, dtype=dtype)

    def call(self, element):
        if element.dtype != self._dtype_holder.dtype:
            return torch.Tensor.to(element, dtype=self._dtype_holder.dtype)
        else:
            return element
