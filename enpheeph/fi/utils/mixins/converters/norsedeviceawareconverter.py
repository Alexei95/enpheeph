import copy

import enpheeph.fi.utils.mixins.converters.norseconverter
import enpheeph.fi.utils.mixins.converters.pytorchdeviceawareconverter


class NorseDeviceAwareConverter(
        enpheeph.fi.utils.mixins.converters.norseconverter.NorseConverter,
        enpheeph.fi.utils.mixins.converters.
        pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    pass
