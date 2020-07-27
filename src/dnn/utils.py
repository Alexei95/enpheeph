import copy
import pathlib

import torch

from . import train
from ..common import USE_CUDA

def init_model(model_class, model_args, *, use_saved_model=True, model_save_path=None, cuda=USE_CUDA, train_dataset=None, optimizer_class=None, optimizer_args=None, loss=None, n_epochs=None):
    model = model_class(**model_args)
    if use_saved_model and model_save_path is not None:
        try:
            new_model = load_model(model, model_save_path, cuda=cuda)
        except Exception as e:
            new_model = None
        else:
            # FIXME: improve this dict return
            new_model = {'model': model, 'accuracy': float('nan'), 'loss': float('nan')}
            return new_model

    if new_model is None or not use_saved_model and train_dataset is not None and optimizer_class is not None and loss is not None and n_epochs is not None:
        return train.train(model, train_dataset=train_dataset, optimizer_class=optimizer_class, loss=loss, n_epochs=n_epochs, cuda=cuda, optimizer_args=optimizer_args)

    # FIXME: improve this Exception
    raise Exception('Invalid arguments, either use_saved_model and model_save_path must not be None or the training args must not be None')


def load_model(base_model, path, *, cuda=USE_CUDA):
    if not pathlib.Path(path).exists():
        raise Exception('path must exist to be loaded')

    model = copy.deepcopy(base_model)

    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model.load_state_dict(torch.load(str(path), map_location=dev))
    model = model.to(dev)

    return model
