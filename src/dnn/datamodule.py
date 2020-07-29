import pytorch_lightning as pl
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from ..common import DEFAULT_DATASET_PATH

TRAIN_STRING = 'train'
TEST_STRING = 'test'
VALIDATION_STRING = 'validation'
DEFAULT_TRANSFORMS = {torchvision.datasets.MNIST.__name__:
                        {TRAIN_STRING:
                            torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ]),
                        TEST_STRING:
                        }
                    }
# this default percentage is 
DEFAULT_DATASET_TRAIN_PERCENTAGE = 1.0

class DataModule(pl.LightningDataModule):
    def __init__(self, path=DEFAULT_DATASET_PATH, train_percentage=DEFAULT_DATASET_TRAIN_PERCENTAGE,)

    def prepare_data(self):
        # download
        MNIST(os.getcwd(), train=True, download=True, transform=None)
        MNIST(os.getcwd(), train=False, download=True, transform=None)

    def setup(self, stage):
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
        # train/val split
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=64)
