import copy

import src.utils.mixins.converters.norseconverter
import src.utils.mixins.converters.pytorchdeviceawareconverter


class NorseDeviceAwareConverter(
        src.utils.mixins.converters.norseconverter.NorseConverter,
        src.utils.mixins.converters.
        pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    pass
