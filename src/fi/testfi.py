import json
import pprint

import torch
import torchvision

import hardwaremodel
import summary

# model
alexnet = torchvision.models.AlexNet()
# input size
alexnet_input_size = torch.Size([1, 3, 224, 224])

s = summary.Summary(alexnet, alexnet_input_size)

pprint.pprint(s)

print(hardwaremodel.SAMPLE_HIERARCHY_JSON)
print(hardwaremodel.SAMPLE_MAP_JSON)

hwmodel = hardwaremodel.HardwareModel(
    json_hierarchy_string=hardwaremodel.SAMPLE_HIERARCHY_JSON,
    json_map_string=hardwaremodel.SAMPLE_MAP_JSON)

pprint.pprint(hwmodel.hierarchy)
print(hwmodel.hierarchy_json)
pprint.pprint(hwmodel.map)
