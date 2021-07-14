import functools
import itertools
import operator

import more_itertools
import pytest

import src.fi.injection.faultdescriptor
import src.fi.utils.enums.bitvalue
import src.fi.utils.enums.parametertype


ACTIVATION_TYPE = src.fi.utils.enums.parametertype.ParameterType.Activation
SNN_STATE_TYPE = (
        src.fi.utils.enums.parametertype.ParameterType.SNN | src.fi.utils.
        enums.parametertype.ParameterType.State
)


class TestFaultDescriptor:
    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    ACTIVATION_TYPE,
                    id=repr(ACTIVATION_TYPE),
            ),
            pytest.param(
                    SNN_STATE_TYPE,
                    id=repr(SNN_STATE_TYPE),
            ),
        ]
    )
    @pytest.mark.parametrize(
        'bit_value',
        [
            pytest.param(
                bit_value,
                id=bit_value.name
            )
            for bit_value in src.fi.utils.enums.bitvalue.BitValue
        ]
    )
    def test_init_exceptions_parameter_name_missing_but_valid(
            self,
            parameter_type,
            bit_value
    ):
        # here we have a missing parameter name
        # it can be omitted only for SNN state or activation injection
        assert src.fi.injection.faultdescriptor.FaultDescriptor(
                module_name='',
                parameter_type=parameter_type,
                tensor_index=[],
                bit_index= [],
                bit_value=bit_value,
        ).parameter_name is None

    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    parameter_type,
                    id=parameter_type.name,
            )
            for parameter_type in (
                    src.fi.utils.enums.parametertype.ParameterType
            )
            if (
                    ACTIVATION_TYPE not in parameter_type and \
                    SNN_STATE_TYPE not in parameter_type
            )
        ]
    )
    @pytest.mark.parametrize(
        'bit_value',
        [
            pytest.param(
                bit_value,
                id=bit_value.name
            )
            for bit_value in src.fi.utils.enums.bitvalue.BitValue
        ]
    )
    def test_init_exceptions_parameter_name_missing_invalid(
            self,
            parameter_type,
            bit_value
    ):
        with pytest.raises(ValueError):
            # here we have a missing parameter name
            # it can be omitted only for SNN state or activation injection
            src.fi.injection.faultdescriptor.FaultDescriptor(
                    module_name='',
                    parameter_type=parameter_type,
                    tensor_index=[],
                    bit_index= [],
                    bit_value=bit_value,
            )
