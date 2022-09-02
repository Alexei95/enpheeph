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

import typing

import torchinfo

import enpheeph.helpers.summaries.abc.modelsummaryabc
import enpheeph.helpers.summaries

class ModelSummaryTorchinfo(
    enpheeph.helpers.summaries.abc.modelsummaryabc.ModelSummaryABC
):
    layers: typing.Optional[typing.List[enpheeph.helpers.summaries.LayerSummaryTorchinfo]]
    def __init__(self, sensitivity_analysis_plugin=None):
        self.sensitivity_analysis_plugin = sensitivity_analysis_plugin
        self.layers = None
        self.summary = None

    def gather_summary(self, model, input_size):
        self.summary = torchinfo.summary(
            # batch_dim is a bit counterintuitive, as it refers to the
            # position of the batch size, not the actual batch size
            model=model, input_size=input_size, batch_dim=0, verbose=0
        )

    def run_analysis(self, model, test_input):
        if self.sensitivity_analysis_plugin is None:
            return
        self.sensitivity_analysis_plugin.run_analysis(model, test_input, layers=self.layers)

    def compute_layer_set(self):
        # we gather only the leaf layers, which are the actually being executed
        # NOTE: this limits us to use only these layer names in the automated
        # fault models, while the manual ones don't have any issues as the MUTs still
        # use these in-between containers
        layers = []
        # we filter only the leaf layers by checking the corresponding flag
        leaf_layers = [l for l in self.summary.summary_list if l.is_leaf_layer]
        # then we remove all the layers without any output size, which means these
        # layers are not part of the model but they are mostly utilities, e.g.
        # accuracy/loss computations
        pruned_leaf_layers = [l for l in leaf_layers if l.output_size != []]
        for layer in pruned_leaf_layers:
            parsed_layer = enpheeph.helpers.summaries.layersummarytorchinfo.LayerSummaryTorchinfo(input_size=layer.input_size, weight_size=layer.kernel_size, output_size=layer.output_size, module=layer.module)
            # to determine the correct ordering, we assume the ordering of torchinfo is
            # correct, so the previous index layer is the input and the following one
            # is the output
            try:
                previous_layer = layers[-1]
            except IndexError:
                parsed_layer.set_input_layer(None)
            else:
                if previous_layer.output_size == parsed_layer.input_size or parsed_layer.input_size in previous_layer.output_size:
                    previous_layer.set_output_layer(parsed_layer)
                    parsed_layer.set_input_layer(previous_layer)
                else:
                    raise ValueError("layers should be compatible in terms of input/output")
        else:
            try:
                l = layers[-1]
            except IndexError:
                pass
            else:
                l.set_output_layer(None)

        self.layers = layers
