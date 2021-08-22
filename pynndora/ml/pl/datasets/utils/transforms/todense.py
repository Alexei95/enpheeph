import torch

import src.ml.pl.datasets.utils.transforms.transformabc


class ToDense(src.ml.pl.datasets.utils.transforms.transformabc.TransformABC):
    def call(self, element):
        if element.is_sparse:
            return torch.Tensor.to_dense(element)
        else:
            return element
