import copy
import typing

import enpheeph.fi.utils.mixins.converters.pytorchconverter


class NorseConverter(
        enpheeph.fi.utils.mixins.converters.pytorchconverter.PyTorchConverter
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

    @classmethod
    def get_norse_sequence_time_step_from_index(
            cls,
            index: typing.Sequence[typing.Sequence[typing.Union[int, slice]]]
    ) -> typing.Tuple[int]:
        # FIXME: for now we return the first index
        # it could be customizable
        return tuple(index[0])
