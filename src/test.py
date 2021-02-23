import sys
sys.path.append('../../data/cifar10_pretrained/')

import torch
import torchvision

from fi import basefaultdescriptor, baseinjectioncallback

import vgg

dataset_data = '../../data'
model_data = '../../data/cifar10_pretrained/vgg11_bn.pt'

transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]])
cifar10_test = torchvision.datasets.CIFAR10(root='../../data', download=True, train=False, transform=transform)
cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_set, batch_size=1, shuffle=True, num_workers=4)
vgg_model = vgg.vgg11_bn(pretrained=False)
vgg_model.load_state_dict(model_data)
vgg_model.eval()

for b, batch in enumerate(test_dataset):
