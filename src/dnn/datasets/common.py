import torch
import torchvision.transforms

DEFAULT_TRANSFORM = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                    ])

DEFAULT_TRAIN_TRANSFORM = DEFAULT_TRANSFORM
DEFAULT_TRAIN_NORMALIZE = True
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_TRAIN_PERCENTAGE = 0.9
DEFAULT_TRAIN_SHUFFLE = True
DEFAULT_TRAIN_NUM_WORKERS = 1
DEFAULT_TRAIN_PIN_MEMORY = True and torch.cuda.is_available()
DEFAULT_TRAIN_DROP_INCOMPLETE_BATCH = False
DEFAULT_TRAIN_BATCH_TIMEOUT = 0

DEFAULT_VALIDATION_TRANSFORM = DEFAULT_TRANSFORM
DEFAULT_VALIDATION_NORMALIZE = True
DEFAULT_VALIDATION_BATCH_SIZE = 128
DEFAULT_VALIDATION_PERCENTAGE = 0.1
DEFAULT_VALIDATION_SHUFFLE = False
DEFAULT_VALIDATION_NUM_WORKERS = 1
DEFAULT_VALIDATION_PIN_MEMORY = True and torch.cuda.is_available()
DEFAULT_VALIDATION_DROP_INCOMPLETE_BATCH = False
DEFAULT_VALIDATION_BATCH_TIMEOUT = 0

DEFAULT_TEST_TRANSFORM = DEFAULT_TRANSFORM
DEFAULT_TEST_NORMALIZE = True
DEFAULT_TEST_BATCH_SIZE = 128
DEFAULT_TEST_PERCENTAGE = 1.0
DEFAULT_TEST_SHUFFLE = False
DEFAULT_TEST_NUM_WORKERS = 1
DEFAULT_TEST_PIN_MEMORY = True and torch.cuda.is_available()
DEFAULT_TEST_DROP_INCOMPLETE_BATCH = False
DEFAULT_TEST_BATCH_TIMEOUT = 0
