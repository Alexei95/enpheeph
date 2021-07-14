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

MISSING_PARAMETER_NAME_NO_ERROR_TYPES = (
    ACTIVATION_TYPE,
    SNN_STATE_TYPE,
)

# we test only the conversions in the class, as well as the hashability
# but we do not test the dataclasss
class TestFaultDescriptor:
    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    parameter_type,
                    id=repr(parameter_type),
            )
            for parameter_type in MISSING_PARAMETER_NAME_NO_ERROR_TYPES
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
        # so we check it is None for those types, which is the default value
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
            if all(
                    no_error_type not in parameter_type
                    for no_error_type in MISSING_PARAMETER_NAME_NO_ERROR_TYPES
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

    def test_init_bit_index(self):
        assert 0

    def test_init_tensor_index(self):
        assert 0

    def test_hashability(self):
        assert 0

    def test_bit_index_coversion(self):
        assert 0

    def test_tensor_index_coversion(self):
        assert 0
