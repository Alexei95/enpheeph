import copy
import pprint
import pathlib
import sys

import torch
import torchvision

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
enpheeph_PARENT_DIR = (CURRENT_DIR / '..').resolve()

sys.path.append(str(enpheeph_PARENT_DIR))

import enpheeph.fi.model.hardwaremodel
import enpheeph.fi.modeling.nnmodelsummary


# we implement a test model which is suitable for our small test case
class LeNet5(torch.nn.Module):
    def __init__(self, original=True):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)
        if original:
            self.fc1 = torch.nn.Linear(256, 120)
            self.relu3 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(120, 84)
            self.relu4 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(84, 10)
            self.relu5 = torch.nn.ReLU()
        else:
            self.fc1 = torch.nn.Linear(256, 250)
            self.relu3 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(250, 200)
            self.relu4 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(200, 10)
            self.relu5 = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


# model
model = torchvision.models.AlexNet()
# model = LeNet5(original=False)
# input size
model_input_size = torch.Size([1, 3, 224, 224])
# model_input_size = torch.Size([1, 1, 28, 28])

model_summary = enpheeph.fi.modeling.nnmodelsummary.NNModelSummary(model, model_input_size)

# pprint.pprint(model_summary, indent=4)
# pprint.pprint(model_summary.layer_stats, indent=4)

# print(enpheeph.fi.model.hardwaremodel.SAMPLE_HIERARCHY_JSON, indent=4)
# print(enpheeph.fi.model.hardwaremodel.SAMPLE_MAP_JSON, indent=4)

hwmodel = enpheeph.fi.model.hardwaremodel.HardwareModel(
    json_hierarchy_string=enpheeph.fi.model.hardwaremodel.SAMPLE_HIERARCHY_JSON,
    json_map_string=enpheeph.fi.model.hardwaremodel.SAMPLE_MAP_JSON)

# pprint.pprint(hwmodel.hierarchy, indent=4)
# print(hwmodel.hierarchy_json)
# pprint.pprint(hwmodel.map, indent=4)

# here we reduce the model summary
# model_summary = copy.deepcopy(model_summary)
# model_summary._layer_stats = model_summary._layer_stats[:1]

schedule = hwmodel.schedule_model_inference_run(
                model_summary=model_summary,
                target=enpheeph.fi.model.hardwaremodel.NvidiaGPUComponentEnum.CUDACore,
                )

# pprint.pprint(schedule, indent=4)

# pprint.pprint([(t.start_time, t.stop_time) for t in schedule[0]])
