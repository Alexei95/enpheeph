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

local defaults = import "../defaults.libsonnet";
local utils = import "../utils.libsonnet";

{
    local config = self,

    dataset_batch_size:: defaults.dataset_batch_size,
    dataset_name:: null,
    dataset_num_classes:: null,
    dataset_num_workers:: defaults.dataset_num_workers,
    dataset_root:: defaults.dataset_root,
    dataset_subcommand_name:: null,
    dataset_val_split:: defaults.dataset_val_split,

    dataset_complete_dir:: utils.joinPath(self.dataset_root, self.dataset_name),
}
