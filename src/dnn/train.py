import copy
import pathlib

import torch

CUDA_IS_AVAILABLE = torch.cuda.is_available()
USE_CUDA = True and CUDA_IS_AVAILABLE


def train(model, train_dataset, optimizer_class, loss, n_epochs, cuda=USE_CUDA, optimizer_args={}):
    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(dev)

    model.train()

    optim = optimizer_class(model.parameters(), **optimizer_args)

    acc = 0
    avg_loss = 0
    n_elements = 0

    for epoch in range(1, n_epochs + 1):
        for b, batch in enumerate(train_dataset, start=1):
            res = train_step(model, batch, optimizer=optim, loss=loss, cuda=cuda)
            acc += res['accuracy']
            avg_loss += res['loss']
            n_elements += res['n_elements']

    return {'accuracy': acc / n_elements, 'loss': avg_loss / n_elements, 'model': model}


def train_step(model, batch, optimizer, loss, cuda=USE_CUDA):
    data, target = batch

    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(dev)
    data = data.to(dev)
    target = target.to(dev)

    model.train()

    optimizer.zero_grad()

    predictions = model(data)

    computed_loss = loss(predictions, target)
    accuracy = torch.sum(target == torch.argmax(predictions, dim=-1))
    n_elements = target.size()[-1]

    computed_loss.backward()

    optimizer.step()

    return {'accuracy': accuracy.item(), 'loss': computed_loss.item(), 'n_elements': n_elements}
