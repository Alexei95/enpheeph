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

cd "${BASH_SOURCE%/*}/" || exit  # cd into the bundle and use relative paths

jsonnet --ext-str dataset_name=carla --ext-code enable_pruning=false --ext-code enable_qat=false --ext-str model_backbone=mobilenetv3_large_100 --ext-str model_head=fpn --ext-str model_task=semantic_segmentation --ext-str dataset_root=/shared/ml/datasets/vision/ --ext-str dataset_batch_size=16 ./base_task.jsonnet -o ./semantic_segmenter_mobilenetv3_large_100_fpn_carla_false_false.json
