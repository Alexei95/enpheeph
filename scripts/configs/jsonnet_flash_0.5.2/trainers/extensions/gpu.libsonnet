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

local defaults = import "../../defaults.libsonnet";

{
    local config = self,
    trainer_gpu_devices+:: defaults.trainer_gpu_devices,

    "trainer"+: {
        "accelerator": "gpu",
        # we cannot join with +: as the default is null in trainer,
        # so it cannot be join-overridden
        "devices": std.objectValues(config.trainer_gpu_devices),
        "move_metrics_to_cpu": false,
    },
}
