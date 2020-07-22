import torch

# using this fault injector for changing all the 
class RandomTensorFI(torch.nn.Module):
    def __init__(self, enableFI=True):
        super().__init__()
        self.enableFI = enableFI

    def forward(self, x):
        if self.enableFI:
            r = torch.rand_like(x)
        else:
            r = x
        return r

FAULT_INJECTOR_NAME = RandomTensorFI.__name__
