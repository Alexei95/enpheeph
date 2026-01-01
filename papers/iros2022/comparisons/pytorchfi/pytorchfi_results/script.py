# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorchfi.core import fault_injection as pfi_core

import datetime
import random


class AlexNet(nn.Module):
    """
    AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
    Without BN, the start learning rate should be 0.01
    (c) YANG, Wei
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    """
    AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


class Custom_Sampler(torch.utils.data.Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def _get_custom_sampler(singleIndex, total):
    indices = random.choices([singleIndex], k=total)
    return Custom_Sampler(indices)


def main(reps=100):
    torch.manual_seed(0)

    batchsize = 10000
    workers = 1
    channels = 3
    img_size = 32

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="/shared/ml/datasets/vision/CIFAR10/",
        train=False,
        download=True,
        transform=transform,
    )

    custom_sampler = _get_custom_sampler(0, batchsize)

    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=workers,
        sampler=custom_sampler,
    )

    model = alexnet(num_classes=10)

    golden_times = []
    for _i in range(reps):
        model.eval().cuda()
        golden_outputs = []
        time_now = datetime.datetime.utcnow()

        with torch.no_grad():
            for imgs, _label in iter(val_loader):
                imgs = imgs.cuda()
                golden_outputs.append(model(imgs))

        print(f"Golden Time Execution: {datetime.datetime.utcnow() - time_now}")
        # print(len(golden_outputs))
        # print(golden_outputs[0].shape)

        golden_times.append(str(datetime.datetime.utcnow() - time_now))

    batch_i = list(range(batchsize))
    layer_i = [0] * batchsize
    c_i = [0] * batchsize
    h_i = [1] * batchsize
    w_i = [1] * batchsize
    inj_value_i = [10000.0] * batchsize

    inj = pfi_core(
        model,
        batchsize,
        input_shape=[channels, img_size, img_size],
        use_cuda=True,
    )

    corrupt_times = []
    for _i in range(reps):
        corrupt_outputs = []
        time_now = datetime.datetime.utcnow()

        with torch.no_grad():
            for imgs, _label in iter(val_loader):
                corrupt_model = inj.declare_neuron_fi(
                    batch=batch_i,
                    layer_num=layer_i,
                    dim1=c_i,
                    dim2=h_i,
                    dim3=w_i,
                    value=inj_value_i,
                )
                corrupt_model.eval().cuda()
                imgs = imgs.cuda()
                corrupt_outputs.append(corrupt_model(imgs))

        print(f"Corrupt Time Execution: {datetime.datetime.utcnow() - time_now}")
        # print(len(corrupt_outputs))
        # print(corrupt_outputs[0].shape)

        corrupt_times.append(str(datetime.datetime.utcnow() - time_now))

    counter = 0
    for g_out, c_out in zip(golden_outputs, corrupt_outputs):
        if torch.all(c_out.eq(g_out)):
            counter += 1
    # print(f"Correct: {counter / len(golden_outputs)}")

    print("golden," + ",".join(golden_times))
    print("corrupt," + ",".join(corrupt_times))


if __name__ == "__main__":
    main(reps=100)
