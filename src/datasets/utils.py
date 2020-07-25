import pathlib

import torchvision.datasets
import torch.utils.data

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

DEFAULT_DATASET_PATH = PROJECT_DIR / 'datasets'

def train_loader(dataset, batch_size, path=None, shuffle=True, num_workers=1, download=True, transform=None):
    
    training_set = dataset(str(path), train=True, download=download, transform=transform)
    
    training_data_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return training_data_loader

def test_loader(dataset, batch_size, path=None, shuffle=False, num_workers=1, download=True, transform=None):
    # testing set has shuffle=False to allow reproducibility
    testing_set = dataset(str(path), train=False, download=download, transform=transform)
    
    testing_data_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return testing_data_loader
