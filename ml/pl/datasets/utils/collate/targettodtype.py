import torch

import src.ml.pl.datasets.utils.collate.collateabc


class TargetToDtype(src.ml.pl.datasets.utils.collate.collateabc.CollateABC):
    def __init__(self, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # this is a workaround to YAML-serialize the dtype
        # the dtype itself cannot be serialized in YAML, and it is required
        # as the transform may be a hparam to be serialized
        self._dtype_holder = torch.zeros(0, dtype=dtype)

    def call(self, input_, target):
        if target.dtype != self._dtype_holder.dtype:
            return input_, torch.Tensor.to(
                    target,
                    dtype=self._dtype_holder.dtype
            )
        else:
            return input_, target
