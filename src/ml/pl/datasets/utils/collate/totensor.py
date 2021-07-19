import collections
import torch


class ToTensor(object):
    def __call__(self, batch, *args, **kwargs):
        batch_in, batch_target = batch
        return torch.as_tensor(batch_in), torch.as_tensor(batch_target)
