import torch

import src.ml.pl.datasets.utils.transforms.transformabc


class ToDtype(src.ml.pl.datasets.utils.transforms.transformabc.TransformABC):
    def __init__(self, dtype):
        super().__init__()

        self._dtype = dtype

    def call(self, element):
        return torch.Tensor.to(element, dtype=self._dtype)
