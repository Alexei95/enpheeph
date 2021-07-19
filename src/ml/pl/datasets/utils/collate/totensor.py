import torch

import src.ml.pl.datasets.utils.collate.collateabc


class ToTensor(src.ml.pl.datasets.utils.collate.collateabc.CollateABC):
    def call(self, input_, target):
        return torch.as_tensor(input_), torch.as_tensor(target)
