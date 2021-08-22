import torch

import src.ml.pl.datasets.utils.transforms.transformabc


class Permute(src.ml.pl.datasets.utils.transforms.transformabc.TransformABC):
    def __init__(self, *dims):
        super().__init__()

        self.dims = dims

    def call(self, element):
        if element.is_sparse:
            return element.to_dense().permute(*self.dims).to_sparse()
        else:
            return torch.Tensor.permute(element, *self.dims)
