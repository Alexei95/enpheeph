import copy
import pathlib

import torch

CUDA_IS_AVAILABLE = torch.cuda.is_available()
USE_CUDA = True and CUDA_IS_AVAILABLE


def test(model, test_dataset, loss, cuda=USE_CUDA):
    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(dev)

    model.eval()

    acc = 0
    avg_loss = 0
    n_elements = 0

    for b, batch in enumerate(test_dataset, start=1):
        res = test_step(model, batch, loss=loss)
        acc += res['accuracy']
        avg_loss += res['loss']
        n_elements += res['n_elements']

    return {'accuracy': acc / n_elements, 'loss': avg_loss / n_elements}


def test_step(model, batch, loss, cuda=USE_CUDA):
    data, target = batch

    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(dev)
    data = data.to(dev)
    target = target.to(dev)

    model.eval()

    with torch.no_grad():

        predictions = model(data)
        computed_loss = loss(predictions, target)
        accuracy = torch.sum(target == torch.argmax(predictions, dim=-1))
        n_elements = target.size()[-1]

    return {'accuracy': accuracy.item(), 'loss': computed_loss.item(), 'n_elements': n_elements}
