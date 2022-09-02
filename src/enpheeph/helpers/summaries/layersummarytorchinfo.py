# -*- coding: utf-8 -*-
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

import enpheeph.helpers.summaries.abc.layersummaryabc


class LayerSummaryTorchinfo(
    enpheeph.helpers.summaries.abc.layersummaryabc.LayerSummaryABC
):
    def __init__(self, input_size, weight_size, output_size, module):
        self.input_size = input_size
        self.weight_size = weight_size
        self.output_size = output_size
        self.module = module

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

    def set_output_layer(self, output_layer):
        self.output_layer = output_layer

    def set_sensitivity_analysis_result(self, result):
        self.sensitivity_analysis_result = result
