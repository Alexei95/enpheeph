from . import basefi

# using this fault injector for changing all the elements in a tensor with
# random numbers
class RandomTensorFI(basefi.BaseFI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.fi_enabled:
            r = self.element_sampler.torch_sample(low=0, high=1, shape=x.size(), dtype=x.dtype, device=x.device)
        else:
            r = x
        return r

FAULT_INJECTOR = {RandomTensorFI.__name__: RandomTensorFI}
