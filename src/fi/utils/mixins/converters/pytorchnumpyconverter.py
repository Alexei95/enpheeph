import copy

import torch


class PyTorchNumpyConverter(object):
    @classmethod
    def single_pytorch_to_numpy(cls, element: torch.Tensor) -> 'numpy.ndarray':
        # we check we have only 1 element, with empty shape
        if element.nelement() != 1:
            raise ValueError('There must be only 1 element in the array')

        return element.squeeze().cpu().numpy()

    @classmethod
    def pytorch_to_numpy(cls, element: torch.Tensor) -> 'numpy.ndarray':
        return element.squeeze().cpu().numpy()

    # we convert from numpy to torch tensor, with optional device and type
    @classmethod
    def numpy_to_pytorch(
            cls,
            element: 'numpy.ndarray',
            *,
            dtype: torch.tensortype = None,
            device: torch.device = None,
    ) -> torch.Tensor:
        return torch.from_numpy(element).to(device=device, dtype=dtype)
