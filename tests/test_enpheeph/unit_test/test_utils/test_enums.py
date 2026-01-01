# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
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


import enum
import operator

import pytest

import enpheeph.utils.enums


class TestEnums(object):
    def test_bit_fault_value(self):
        assert issubclass(enpheeph.utils.enums.BitFaultValue, enum.Enum)

        assert {"BitFlip", "StuckAtZero", "StuckAtOne"} == set(
            enpheeph.utils.enums.BitFaultValue.__members__.keys()
        )

    def test_bit_width_int_enum(self):
        assert issubclass(enpheeph.utils.enums.BitWidth, enum.IntEnum)

    @pytest.mark.parametrize(
        argnames=("width_name", "bit_width_value"),
        argvalues=[
            pytest.param(
                width_name,
                bit_width_value,
                id=str(width_name) + "_" + str(bit_width_value),
            )
            for width_name, bit_width_value in {
                "OneByte": 8,
                "TwoBytes": 16,
                "ThreeBytes": 24,
                "FourBytes": 32,
                "FiveBytes": 40,
                "SixBytes": 48,
                "SevenBytes": 56,
                "EightBytes": 64,
                "FloatingPoint16": 16,
                "FloatingPoint32": 32,
                "FloatingPoint64": 64,
                "Int32": 32,
                "Int64": 64,
            }.items()
        ],
    )
    def test_bit_width_values(self, width_name, bit_width_value):
        assert (
            enpheeph.utils.enums.BitWidth.__members__[width_name].value
            == bit_width_value
        )

    def test_dimension_type(self):
        assert issubclass(enpheeph.utils.enums.DimensionType, enum.Enum)

        assert {"BitLevel", "Batch", "Tensor", "Time"} == set(
            enpheeph.utils.enums.DimensionType.__members__.keys()
        )

    def test_endianness_enum(self):
        assert issubclass(enpheeph.utils.enums.Endianness, enum.Enum)

    @pytest.mark.parametrize(
        argnames=("endianness_name", "endianness_symbol"),
        argvalues=[
            pytest.param(
                endianness_name,
                endianness_symbol,
                id=str(endianness_name) + "_" + str(endianness_symbol),
            )
            for endianness_name, endianness_symbol in {
                "Little": "<",
                "Big": ">",
                "MSBAtIndexZero": ">",
                "LSBAtIndexZero": "<",
            }.items()
        ],
    )
    def test_endianness_values(self, endianness_name, endianness_symbol):
        assert (
            enpheeph.utils.enums.Endianness.__members__[endianness_name].value
            == endianness_symbol
        )

    def test_fault_mask_operation_enum(self):
        assert issubclass(enpheeph.utils.enums.FaultMaskOperation, enum.Enum)

    @pytest.mark.parametrize(
        argnames=("fault_mask_operation_name", "fault_mask_operation_symbol"),
        argvalues=[
            pytest.param(
                fault_mask_operation_name,
                fault_mask_operation_symbol,
                id=str(fault_mask_operation_name)
                + "_"
                + str(fault_mask_operation_symbol),
            )
            for fault_mask_operation_name, fault_mask_operation_symbol in {
                "InPlaceXor": operator.ixor,
                "InPlaceAnd": operator.iand,
                "InPlaceOr": operator.ior,
                "Xor": operator.xor,
                "And": operator.and_,
                "Or": operator.or_,
            }.items()
        ],
    )
    def test_fault_mask_operation_values(
        self, fault_mask_operation_name, fault_mask_operation_symbol
    ):
        assert (
            enpheeph.utils.enums.FaultMaskOperation.__members__[
                fault_mask_operation_name
            ].value
            == fault_mask_operation_symbol
        )

    def test_fault_mask_value_enum(self):
        assert issubclass(enpheeph.utils.enums.FaultMaskValue, enum.IntEnum)

    @pytest.mark.parametrize(
        argnames=("fault_mask_value_name", "fault_mask_value_symbol"),
        argvalues=[
            pytest.param(
                fault_mask_value_name,
                fault_mask_value_symbol,
                id=str(fault_mask_value_name) + "_" + str(fault_mask_value_symbol),
            )
            for fault_mask_value_name, fault_mask_value_symbol in {
                "One": 1,
                "Zero": 0,
            }.items()
        ],
    )
    def test_fault_mask_value_values(
        self, fault_mask_value_name, fault_mask_value_symbol
    ):
        assert (
            enpheeph.utils.enums.FaultMaskValue.__members__[fault_mask_value_name].value
            == fault_mask_value_symbol
        )

    def test_handler_status(self):
        assert issubclass(enpheeph.utils.enums.HandlerStatus, enum.Enum)

        assert {"Running", "Idle"} == set(
            enpheeph.utils.enums.HandlerStatus.__members__.keys()
        )

    def test_monitor_metric(self):
        assert issubclass(enpheeph.utils.enums.MonitorMetric, enum.Flag)

        assert {
            "StandardDeviation",
            "Maximum",
            "Minimum",
            "ArithmeticMean",
            "GeometricMean",
        } == set(enpheeph.utils.enums.MonitorMetric.__members__.keys())

    def test_parameter_type(self):
        assert issubclass(enpheeph.utils.enums.ParameterType, enum.Flag)

        # we use subset as there are extra items which we check later
        assert {
            "DNN",
            "SNN",
            "RNN",
            "Weight",
            "Activation",
            "State",
            "LIF",
            "Voltage",
            "Current",
            "Dense",
            "PrunedDense",
            "Sparse",
            "COO",
            "CSR",
            "Index",
            "Value",
            "DNNWeightDense",
            "DNNActivationDense",
            "SNNLIFStateVoltageDense",
        } == set(enpheeph.utils.enums.ParameterType.__members__.keys())

    @pytest.mark.parametrize(
        argnames=("parameter_type_composite", "parameter_type_equivalence"),
        argvalues=[
            pytest.param(
                parameter_type_composite,
                parameter_type_equivalence,
                id=str(parameter_type_composite)
                + "_"
                + str(parameter_type_equivalence),
            )
            for parameter_type_composite, parameter_type_equivalence in {
                enpheeph.utils.enums.ParameterType.DNNWeightDense: (
                    enpheeph.utils.enums.ParameterType.DNN
                    | enpheeph.utils.enums.ParameterType.Weight
                    | enpheeph.utils.enums.ParameterType.Dense
                ),
                enpheeph.utils.enums.ParameterType.DNNActivationDense: (
                    enpheeph.utils.enums.ParameterType.DNN
                    | enpheeph.utils.enums.ParameterType.Activation
                    | enpheeph.utils.enums.ParameterType.Dense
                ),
                enpheeph.utils.enums.ParameterType.SNNLIFStateVoltageDense: (
                    enpheeph.utils.enums.ParameterType.SNN
                    | enpheeph.utils.enums.ParameterType.LIF
                    | enpheeph.utils.enums.ParameterType.State
                    | enpheeph.utils.enums.ParameterType.Dense
                    | enpheeph.utils.enums.ParameterType.Voltage
                ),
            }.items()
        ],
    )
    def test_parameter_type_composite_values(
        self, parameter_type_composite, parameter_type_equivalence
    ):
        assert parameter_type_composite == parameter_type_equivalence
