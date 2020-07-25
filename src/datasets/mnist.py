import pathlib
import sys

import torchvision.datasets
import torchvision.transforms

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

import src.datasets.utils

TRAIN_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
TEST_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])

def mnist(train_batch_size, test_batch_size, path=None, num_workers=1):
    training = src.datasets.utils.train_loader(torchvision.datasets.MNIST, train_batch_size, path=path, num_workers=num_workers, transform=TRAIN_TRANSFORM)

    testing = src.datasets.utils.test_loader(torchvision.datasets.MNIST, test_batch_size, path=path, num_workers=num_workers, transform=TEST_TRANSFORM)

    return training, testing

DATASET = {'mnist': mnist}
