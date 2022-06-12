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

{
    local config = self,
    model_backbone:: null,
    model_backbone_kwargs+:: {},
    model_base_name:: null,
    model_head:: null,
    model_learning_rate:: null,
    model_loss_fn:: null,
    model_metrics+:: {},
    # this metric is required for the trainer callbacks
    model_monitor_metric:: null,
    model_multi_label:: false,
    model_name:: self.model_base_name + "_" + self.model_specific_name,
    model_num_classes:: null,
    model_optimizer:: "Adam",
    model_specific_name:: self.model_backbone + "_" + self.model_head,
    model_task:: null,
    model_training_strategy:: "default",
    model_training_strategy_kwargs+:: {},
}
