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

local image_classifier = import "./image_classifier.libsonnet";
local object_detector = import "./object_detector.libsonnet";
local semantic_segmenter = import "./semantic_segmenter.libsonnet";

local models = {
    [image_classifier.model_task]: image_classifier,
    [object_detector.model_task]: object_detector,
    [semantic_segmenter.model_task]: semantic_segmenter,
};

function(model_task)
    if std.objectHas(models, model_task) then
        models[model_task]
    else
        error
            "invalid dataset name, choose between the following: " +
            std.join(
                ",",
                std.objectKeys(models),
            )
