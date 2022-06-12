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

local carla = import "./carla.libsonnet";
local cifar10 = import "./cifar10.libsonnet";
local coco = import "./coco.libsonnet";

local datasets = {
    [carla.dataset_name]: carla,
    [cifar10.dataset_name]: cifar10,
    [coco.dataset_name]: coco,
};

function(dataset_name)
    if std.objectHas(datasets, dataset_name) then
        datasets[dataset_name]
    else
        error
            "invalid dataset name, choose between the following: " +
            std.join(
                ",",
                std.objectKeys(datasets),
            )
