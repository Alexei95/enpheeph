import torch

class RandomBitFlipFI(torch.nn.Module):
    def __init__(self, enableFI=True, coverage=1.0, n_elements=None, n_bit_flips=1):
        super().__init__()
        self.enableFI = enableFI
        if (coverage is None and n_elements is None) or (coverage is not None and n_elements is not None):
            raise Exception('Only one of coverage and n_elements must be different from None')
        self.coverage = coverage
        self.n_elements = n_elements
        self.n_bit_flips = n_bit_flips  # tailored for fp32, can be changed

    def forward(self, x):
        if self.enableFI:
            # total = functools.reduce(operator.mul, x.size())
            # covered = math.ceil(self.percentage * total)
            # index = [tuple(torch.randint(low=0, high=index_dim, size=(1, )).item() for index_dim in x.size()) for _ in range(covered)]
            # r = torch.clone(x).detach()  # add requires_grad_(True) for grad
            # for i in index:
            #     r[i] = torch.randn(size=(1, ))
            r = torch.clone(x).detach().flatten()  # add requires_grad_(True) for grad
            perm = torch.randperm(r.numel())  # we could use cuda but loop is in Python
            if self.coverage is not None and self.n_elements is None:
                covered = math.ceil(self.coverage * r.numel())
            elif self.coverage is None and self.n_elements is not None:
                covered = self.n_elements
            covered = min(max(0, covered), r.numel())
            for i in range(covered):
                r[perm[i]] = bit_flip(r[perm[i]], self.n_bit_flips)
            r = r.reshape(x.size())
        else:
            r = x
        return r
