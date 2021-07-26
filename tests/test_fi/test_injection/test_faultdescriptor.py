import random

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

    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    parameter_type,
                    id=repr(parameter_type),
            )
            for parameter_type in (
                    src.fi.utils.enums.parametertype.ParameterType
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
    @pytest.mark.parametrize(
        'bit_index',
        [
            pytest.param(
                bit_index,
                id=str(bit_index)
            )
            for bit_index in ([], [slice(10), ..., 1, 10])
        ]
    )
    def test_init_bit_index_tuple_conversion(
            self,
            parameter_type,
            bit_index,
            bit_value
    ):        # it can be omitted only for SNN state or activation injection
        assert src.fi.injection.faultdescriptor.FaultDescriptor(
                module_name='',
                parameter_type=parameter_type,
                parameter_name="",
                tensor_index=[],
                bit_index= bit_index,
                bit_value=bit_value,
        ).bit_index == tuple(bit_index)

    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    parameter_type,
                    id=repr(parameter_type),
            )
            for parameter_type in (
                    src.fi.utils.enums.parametertype.ParameterType
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
    @pytest.mark.parametrize(
        'tensor_index',
        [
            pytest.param(
                tensor_index,
                id=str(tensor_index)
            )
            for tensor_index in ([], [slice(10), ..., 1, 10])
        ]
    )
    def test_init_tensor_index_tuple_conversion(
            self,
            parameter_type,
            tensor_index,
            bit_value
    ):
        assert src.fi.injection.faultdescriptor.FaultDescriptor(
                module_name='',
                parameter_type=parameter_type,
                parameter_name="",
                tensor_index=tensor_index,
                bit_index=[],
                bit_value=bit_value,
        ).tensor_index == tuple(tensor_index)

    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    parameter_type,
                    id=repr(parameter_type),
            )
            for parameter_type in (
                    src.fi.utils.enums.parametertype.ParameterType
            )
        ]
    )
    @pytest.mark.parametrize(
        'tensor_index,correct_tensor_index',
        [
            pytest.param(
                tensor_index,
                (
                        tuple(tensor_index)
                        if isinstance(tensor_index, list)
                        else tensor_index
                ),
                id=str(tensor_index)
            )
            for tensor_index in ([], [slice(10), ..., 1, 10], ...)
        ]
    )
    @pytest.mark.parametrize(
        'bit_index,correct_bit_index',
        [
            pytest.param(
                bit_index,
                tuple(bit_index) if isinstance(bit_index, list) else bit_index,
                id=str(bit_index)
            )
            for bit_index in ([], [slice(10), ..., 1, 10], ...)
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
    def test_init(
            self,
            parameter_type,
            tensor_index,
            correct_tensor_index,
            bit_index,
            correct_bit_index,
            bit_value
    ):
        values = {
                'module_name': (
                        ''.join(
                                str(random.randint(0, n))
                                for n in range(random.randint(0, 100))
                        )
                ),
                'parameter_type': parameter_type,
                'parameter_name': (
                        ''.join(
                                str(random.randint(0, n))
                                for n in range(random.randint(1, 100))
                        )
                ),
                'tensor_index': tensor_index,
                'bit_index': bit_index,
                'bit_value': bit_value,
        }
        fd = src.fi.injection.faultdescriptor.FaultDescriptor(
                **values
        )
        correct_values = values.copy()
        correct_values.update(
                {
                        'bit_index': correct_bit_index,
                        'tensor_index': correct_tensor_index,
                }
        )

        for name, value in correct_values.items():
            assert getattr(fd, name) == value

    @pytest.mark.parametrize(
        'parameter_type',
        [
            pytest.param(
                    parameter_type,
                    id=repr(parameter_type),
            )
            for parameter_type in (
                    src.fi.utils.enums.parametertype.ParameterType
            )
        ]
    )
    @pytest.mark.parametrize(
        'tensor_index',
        [
            pytest.param(
                tensor_index,
                id=str(tensor_index)
            )
            for tensor_index in ([], [slice(10), ..., 1, 10], ...)
        ]
    )
    @pytest.mark.parametrize(
        'bit_index',
        [
            pytest.param(
                bit_index,
                id=str(bit_index)
            )
            for bit_index in ([], [slice(10), ..., 1, 10], ...)
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
    def test_hashability(
            self,
            parameter_type,
            tensor_index,
            bit_index,
            bit_value
    ):
        values_one = {
                'module_name': (
                        ''.join(
                                str(random.randint(0, n))
                                for n in range(random.randint(0, 100))
                        )
                ),
                'parameter_type': parameter_type,
                'parameter_name': (
                        ''.join(
                                str(random.randint(0, n))
                                for n in range(random.randint(0, 100))
                        )
                ),
                'tensor_index': tensor_index,
                'bit_index': bit_index,
                'bit_value': bit_value,
        }
        fd_one = src.fi.injection.faultdescriptor.FaultDescriptor(
                **values_one
        )

        values_two = {
                'module_name': (
                        ''.join(
                                str(random.randint(0, n))
                                for n in range(random.randint(0, 100))
                        )
                ),
                'parameter_type': parameter_type,
                'parameter_name': (
                        ''.join(
                                str(random.randint(0, n))
                                for n in range(random.randint(0, 100))
                        )
                ),
                'tensor_index': tensor_index,
                'bit_index': bit_index,
                'bit_value': bit_value,
        }
        fd_two = src.fi.injection.faultdescriptor.FaultDescriptor(
                **values_two
        )

        assert hash(fd_one) != hash(fd_two)

    @pytest.mark.parametrize(
        'bit_index,bit_width,converted_bit_index',
        [
            pytest.param(
                    [],
                    32,
                    (),
                    id=str(([], 32))
            ),
            pytest.param(
                    slice(0, 10, 2),
                    16,
                    (0, 2, 4, 6, 8),
                    id=str((slice(0, 10, 2), 16))
            ),
            pytest.param(
                    slice(0, 10, 2),
                    7,
                    (0, 2, 4, 6),
                    id=str((slice(0, 10, 2), 7))
            ),
            pytest.param(
                    [0, 2, 10],
                    16,
                    (0, 2, 10),
                    id=str(([0, 2, 10], 16))
            ),
            pytest.param(
                    [0, 2, 10],
                    7,
                    (0, 2),
                    id=str(([0, 2, 10], 7))
            ),
            pytest.param(
                    ...,
                    7,
                    (0, 1, 2, 3, 4, 5, 6),
                    id=str((..., 7))
            ),
            pytest.param(
                    [1, 1, 2, 0, 4, 7, 5, 1, 1, 1, 1, 1],
                    10,
                    (0, 1, 2, 4, 5, 7),
                    id=str(([1, 1, 2, 0, 4, 7, 5, 1, 1, 1, 1, 1], 10))
            ),
        ]
    )
    def test_bit_index_coversion(
            self,
            bit_index,
            bit_width,
            converted_bit_index
    ):
        conv_bit_index = (
                src.fi.injection.faultdescriptor.
                FaultDescriptor.bit_index_conversion(
                        bit_index,
                        bit_width
                )
        )
        assert conv_bit_index == converted_bit_index

    @pytest.mark.parametrize(
        'bit_index,bit_width',
        [
            pytest.param(
                    [],
                    32,
                    (),
                    id=str(([], 32))
            ),
            pytest.param(
                    slice(0, 10, 2),
                    16,
                    (0, 2, 4, 6, 8),
                    id=str((slice(0, 10, 2), 16))
            ),
            pytest.param(
                    slice(0, 10, 2),
                    7,
                    (0, 2, 4, 6),
                    id=str((slice(0, 10, 2), 7))
            ),
            pytest.param(
                    [0, 2, 10],
                    16,
                    (0, 2, 10),
                    id=str(([0, 2, 10], 16))
            ),
            pytest.param(
                    [0, 2, 10],
                    7,
                    (0, 2),
                    id=str(([0, 2, 10], 7))
            ),
            pytest.param(
                    ...,
                    7,
                    (0, 1, 2, 3, 4, 5, 6),
                    id=str((..., 7))
            ),
            pytest.param(
                    [1, 1, 2, 0, 4, 7, 5, 1, 1, 1, 1, 1],
                    10,
                    (0, 1, 2, 4, 5, 7),
                    id=str(([1, 1, 2, 0, 4, 7, 5, 1, 1, 1, 1, 1], 10))
            ),
        ]
    )
    def test_bit_index_coversion_value_error(
            self,
            bit_index,
            bit_width
    ):
        with pytest.raises(ValueError):
            (
                    src.fi.injection.faultdescriptor.
                    FaultDescriptor.bit_index_conversion(
                            bit_index,
                            bit_width
                    )
            )

    @pytest.mark.skip("Not ready yet")
    @pytest.mark.parametrize(
        'tensor_index,tensor_shape,converted_tensor_index',
        [
            pytest.param(
                    [],
                    32,
                    (),
                    id=str(([], 32))
            ),
            pytest.param(
                    slice(0, 10, 2),
                    16,
                    (0, 2, 4, 6, 8),
                    id=str((slice(0, 10, 2), 16))
            ),
            pytest.param(
                    slice(0, 10, 2),
                    7,
                    (0, 2, 4, 6),
                    id=str((slice(0, 10, 2), 7))
            ),
            pytest.param(
                    [0, 2, 10],
                    16,
                    (0, 2, 10),
                    id=str(([0, 2, 10], 16))
            ),
            pytest.param(
                    [0, 2, 10],
                    7,
                    (0, 2),
                    id=str(([0, 2, 10], 7))
            ),
            pytest.param(
                    ...,
                    7,
                    (0, 1, 2, 3, 4, 5, 6),
                    id=str((..., 7))
            ),
            pytest.param(
                    [1, 1, 2, 0, 4, 7, 5, 1, 1, 1, 1, 1],
                    10,
                    (0, 1, 2, 4, 5, 7),
                    id=str(([1, 1, 2, 0, 4, 7, 5, 1, 1, 1, 1, 1], 10))
            ),
        ]
    )
    def test_tensor_index_coversion(
            self,
            tensor_index,
            tensor_shape,
            force_index,
            converted_tensor_index
    ):
        conv_tensor_index = (
                src.fi.injection.faultdescriptor.
                FaultDescriptor.tensor_index_conversion(
                        tensor_index,
                        tensor_shape,
                        force_index
                )
        )
        assert conv_tensor_index == converted_tensor_index
