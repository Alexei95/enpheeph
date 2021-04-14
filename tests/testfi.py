import json
import pprint
import pathlib
import sys

import torch
import torchvision

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_PARENT_DIR = (CURRENT_DIR / '..').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import src.fi.hardwaremodel
import src.fi.summary

# model
alexnet = torchvision.models.AlexNet()
# input size
alexnet_input_size = torch.Size([1, 3, 224, 224])

alexnet_summary = src.fi.summary.Summary(alexnet, alexnet_input_size)

#pprint.pprint(alexnet_summary)
pprint.pprint(alexnet_summary.layer_stats)

#print(src.fi.hardwaremodel.SAMPLE_HIERARCHY_JSON)
#print(src.fi.hardwaremodel.SAMPLE_MAP_JSON)

hwmodel = src.fi.hardwaremodel.HardwareModel(
    json_hierarchy_string=src.fi.hardwaremodel.SAMPLE_HIERARCHY_JSON,
    json_map_string=src.fi.hardwaremodel.SAMPLE_MAP_JSON)

#pprint.pprint(hwmodel.hierarchy)
#print(hwmodel.hierarchy_json)
#pprint.pprint(hwmodel.map)

alexnet_schedule = hwmodel.schedule_model_inference_run(
                model_summary=alexnet_summary,
                target=src.fi.hardwaremodel.NvidiaGPUComponentEnum.CUDACore,
                )

pprint.pprint(
    (len(alexnet_schedule),
     tuple(len(d) for d in alexnet_schedule.values()))
    )
