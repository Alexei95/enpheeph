import pathlib
import sys

import torchvision.datasets
import torchvision.transforms

# PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
# if str(PROJECT_DIR) not in sys.path:
#     sys.path.append(str(PROJECT_DIR))

from . import utils

TRAIN_TRANSFORM = torchvision.transforms.Compose([
                                        torchvision.transforms.RandomCrop(32, padding=4),
                                        torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                        ])
TEST_TRANSFORM = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                    ])

def CIFAR10(train_batch_size, test_batch_size, path=None, num_workers=1):
    training = utils.train_loader(torchvision.datasets.CIFAR10, train_batch_size, path=path, num_workers=num_workers, transform=TRAIN_TRANSFORM)

    testing = utils.test_loader(torchvision.datasets.CIFAR10, test_batch_size, path=path, num_workers=num_workers, transform=TEST_TRANSFORM)

    return training, testing

DATASET = {CIFAR10.__name__: CIFAR10}
