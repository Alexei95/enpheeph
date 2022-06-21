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

jsonnet --ext-str dataset_name=cifar10 --ext-code enable_pruning=true --ext-code enable_qat=false --ext-str model_backbone=vgg11 --ext-code model_head=null --ext-str model_task=image_classification --ext-str dataset_root=/shared/ml/datasets/vision/ --ext-str dataset_batch_size=32 ./base_task.jsonnet -o ./image_classifier_vgg11_null_cifar10_true_false.json
