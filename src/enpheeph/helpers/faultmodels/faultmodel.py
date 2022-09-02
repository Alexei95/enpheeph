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

import random

import enpheeph.helpers.faultmodels.abc.faultmodelabc


# for now we support only activations, so layer outputs
class ImportanceSampling(enpheeph.helpers.faultmodels.abc.faultmodelabc.FaultModelABC):
    def __init__(self, model_summary, random_ratio=0.1, seed=42):
        self.model_summary = model_summary
        self.generator = random.Random(seed)
        self.random_ratio = random_ratio

    def get_new_sample(self):
        random_number = self.generator.random()
        if random_number > self.random_ratio:
            return self._get_sensitivity_sample()
        else:
            return self._get_random_sample()

    def _get_random_sample(self):
       layer = self.generator.sample(self.model_summary.layers, 1)[0]
       output_size = layer.output_size
