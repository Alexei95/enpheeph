import copy

import src.utils.mixins.converters.pytorchconverter


class NorseConverter(
        src.utils.mixins.converters.pytorchconverter.PyTorchConverter
):
    @classmethod
    def remove_norse_sequence_time_step_from_shape(
            cls,
            size: "torch.Size"
    ) -> "torch.Size":
        # FIXME: this method should be able to remove a custom dimension
        # or use other way to recognize the correct one to remove, e.g.
        # named tensors or similar other tricks
        # for now we only remove the first dimension
        return size[1:]
