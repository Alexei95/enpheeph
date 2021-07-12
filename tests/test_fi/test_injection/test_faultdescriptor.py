import pytest

import src.fi.injection.faultdescriptor
import src.fi.utils.enums.parametertype

class TestFaultDescriptor:
    def test_init_exceptions_parameter_name_invalid(self):
        with pytest.raises(ValueError):
            src.fi.injection.faultdescriptor.FaultDescriptor(
                    module_name='',
                    parameter_type=src.fi.injection.parametertype.ParameterType.Voltage,
                    tensor_index=[],
                    bit_index= [],
                    bit_value=src.fi.utils.enums.bitvalue.BitValue.BitFlip,
            )
