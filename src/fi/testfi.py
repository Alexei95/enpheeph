import torch
import torchvision

import summary

alexnet = torchvision.models.AlexNet()
alexnet_input_size = torch.Size([1, 3, 224, 224])

s = summary.Summary(alexnet, alexnet_input_size)

print(s.layer_stats)
