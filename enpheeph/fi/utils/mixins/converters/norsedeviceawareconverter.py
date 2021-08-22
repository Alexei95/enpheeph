import copy

import src.fi.utils.mixins.converters.norseconverter
import src.fi.utils.mixins.converters.pytorchdeviceawareconverter


class NorseDeviceAwareConverter(
        src.fi.utils.mixins.converters.norseconverter.NorseConverter,
        src.fi.utils.mixins.converters.
        pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    pass
