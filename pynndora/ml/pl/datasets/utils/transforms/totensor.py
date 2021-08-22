import torch

import src.ml.pl.datasets.utils.transforms.transformabc


class ToTensor(src.ml.pl.datasets.utils.transforms.transformabc.TransformABC):
    def call(self, element):
        return torch.as_tensor(element)
