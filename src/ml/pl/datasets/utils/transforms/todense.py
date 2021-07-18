import torch

import src.ml.pl.datasets.utils.transforms.transformabc


class ToDense(src.ml.pl.datasets.utils.transforms.transformabc.TransformABC):
    def call(self, element):
        return torch.Tensor.to_dense(element)
