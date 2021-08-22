import torch

import src.ml.pl.datasets.utils.collate.collateabc


class TargetToOneHot(src.ml.pl.datasets.utils.collate.collateabc.CollateABC):
    def __init__(self, num_classes=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes

    def call(self, input_, target):
        return input_, torch.nn.functional.one_hot(
                target,
                num_classes=self.num_classes
        )
