# -*- coding: utf-8 -*-
import abc
import typing

import enpheeph.injections.plugins.indexing.indexingpluginabc
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.injections.pytorchinjectionabc
import enpheeph.utils.data_classes
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if typing.TYPE_CHECKING:
    import torch


class PyTorchSparseInterfaceMixin(abc.ABC):
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.indexingpluginabc.IndexingPluginABC
    )
    # the used variables in the functions, must be initialized properly
    location: enpheeph.utils.data_classes.BaseInjectionLocation

    def get_sparse_injection_parameter(
        self,
        tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        sparse_output = tensor.to_sparse()

        # mypy has some issues in recognizing the enum names if taken from a name itself
        # e.g. A.a.a
        # we use separate values to avoid this issue
        # however we still require typing from the enum,
        # which limits the customizability of the interface, as before it could be any
        # compatible enum but now it must be this specific one
        # **NOTE**: a possible alternative is using .value at the end to extract the
        # correct enum, which does nothing
        # however value returns the integer value, so it is still not a clean trick
        sparse_index_flag = (
            self.location.parameter_type.Sparse | self.location.parameter_type.Index
        )
        sparse_value_flag = (
            self.location.parameter_type.Sparse | self.location.parameter_type.Value
        )
        if sparse_index_flag.value in self.location.parameter_type:
            target = sparse_output.indices()
        elif sparse_value_flag.value in self.location.parameter_type:
            target = sparse_output.values()
        else:
            raise ValueError("This operation is not supported with sparse tensors")

        return target
