import torch

CUDA_IS_AVAILABLE = torch.cuda.is_available()
USE_CUDA = True and CUDA_IS_AVAILABLE

def init_model(model, *, use_saved_model=True, model_save_path=None, cuda=USE_CUDA, train_dataset=None, optimizer=None, optimizer_args=None, loss=None, n_epochs=30):
    if use_saved_model:
        model = load_model(model, model_save_path, cuda)
    else:
        model = train()['model']
    return model


def load_model(base_model, path, *, cuda=USE_CUDA):
    if not pathlib.Path(path).exists():
        raise Exception('path must exist to be loaded')

    model = copy.deepcopy(base_model)

    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model.load_state_dict(torch.load(str(path), map_location=dev))
    model = model.to(dev)

    return model



def train(model, train_dataset, optimizer, loss, n_epochs, cuda=USE_CUDA, optimizer_args=None):
    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(device)

    model.train()
    
    optim = optimizer(model.parameters(), **optimizer_args)

    acc = 0
    loss = 0
    n_elements = 0

    for epoch in range(1, epochs + 1):
        for b, batch in enumerate(training_data_loader, start=1):
            res = train_step(model, batch, optimizer=optim, loss=loss, cuda=cuda)
            acc += res['accuracy']
            loss += res['loss']
            n_elements += res['n_elements']

    return {'accuracy': acc / n_elements, 'loss': loss / n_elements, 'model': model}


def train_step(model, batch, optimizer, loss, cuda=USE_CUDA):
    data, target = batch

    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(device)
    data = data.to(device)
    target = target.to(device)

    model.train()

    optim.zero_grad()

    predictions = model(data)
    
    computed_loss = loss(predictions, target)
    accuracy = torch.sum(target == torch.argmax(predictions, dim=-1))
    n_elements = target.size()[-1]
    
    computed_loss.backward()
    
    optim.step()

    return {'accuracy': accuracy.item(), 'loss': computed_loss.item(), 'n_elements': n_elements}


def test(model, test_dataset, loss, cuda=USE_CUDA):
    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(device)

    model.eval()

    acc = 0
    loss = 0
    n_elements = 0

    for b, batch in enumerate(test_dataset, start=1):
        res = test_step(model, batch, loss=loss)
        acc += res['accuracy']
        loss += res['loss']
        n_elements += res['n_elements']

    return {'accuracy': acc / n_elements, 'loss': loss / n_elements}


def test_step(model, batch, loss, cuda=USE_CUDA):
    data, target = batch

    dev = cuda if isinstance(cuda, torch.device) else (torch.device('cuda') if cuda else torch.device('cpu'))
    model = model.to(device)
    data = data.to(device)
    target = target.to(device)

    model.eval()

    with torch.no_grad():

        predictions = model(data)
        computed_loss = loss(predictions, target)
        accuracy = torch.sum(target == torch.argmax(predictions, dim=-1))
        n_elements = target.size()[-1]

    return {'accuracy': accuracy.item(), 'loss': computed_loss.item(), 'n_elements': n_elements}
