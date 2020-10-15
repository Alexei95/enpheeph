import torch
import torchvision.transforms

VISION_DEFAULT_NORMALIZE = True

VISION_DEFAULT_TRANSFORM = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                    ])

VISION_DEFAULT_TRAIN_TRANSFORM = VISION_DEFAULT_TRANSFORM

VISION_DEFAULT_VALIDATION_TRANSFORM = VISION_DEFAULT_TRANSFORM

VISION_DEFAULT_TEST_TRANSFORM = VISION_DEFAULT_TRANSFORM
