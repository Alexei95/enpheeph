# -*- coding: utf-8 -*-
import typing

import enpheeph.injections.monitorabc
import enpheeph.injections.plugins.indexing.indexingpluginabc
import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmonitorpostprocessormixin
import enpheeph.injections.plugins.storage.storagepluginabc
import enpheeph.utils.data_classes
import enpheeph.utils.enums

# so flake does not complain about the imports being not at the top after the if
if typing.TYPE_CHECKING:
    import torch


class OutputPyTorchMonitor(
    enpheeph.injections.monitorabc.MonitorABC,
    enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
    (
        # black has issues with very long names
        # fmt: off
        enpheeph.injections.mixins.
        pytorchmonitorpostprocessormixin.PyTorchMonitorPostProcessorMixin
        # fmt: on
    ),
):
    enabled_metrics: enpheeph.utils.enums.MonitorMetric
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.indexingpluginabc.IndexingPluginABC
    )
    location: enpheeph.utils.data_classes.MonitorLocation
    move_to_first: bool
    storage_plugin: (
        enpheeph.injections.plugins.storage.storagepluginabc.StoragePluginABC
    )

    def __init__(
        self,
        indexing_plugin: (
            enpheeph.injections.plugins.indexing.indexingpluginabc.IndexingPluginABC
        ),
        location: enpheeph.utils.data_classes.MonitorLocation,
        enabled_metrics: enpheeph.utils.enums.MonitorMetric,
        storage_plugin: (
            enpheeph.injections.plugins.storage.storagepluginabc.StoragePluginABC
        ),
        move_to_first: bool = True,
    ):
        super().__init__()

        self.indexing_plugin = indexing_plugin
        self.location = location
        self.enabled_metrics = enabled_metrics
        self.storage_plugin = storage_plugin
        self.move_to_first = move_to_first

        self.handle = None

    @property
    def module_name(self) -> str:
        return self.location.module_name

    # this is compatible with PyTorch hook arguments and return type
    def output_monitor_hook(
        self,
        module: "torch.nn.Module",
        input: typing.Union[typing.Tuple["torch.Tensor"], "torch.Tensor"],
        output: "torch.Tensor",
    ) -> None:
        self.indexing_plugin.select_active_dimensions(
            [
                enpheeph.utils.enums.DimensionType.Batch,
                enpheeph.utils.enums.DimensionType.Tensor,
            ],
            autoshift_to_boundaries=True,
            fill_empty_index=True,
            filler=slice(None, None),
        )
        # NOTE: no support for bit_index yet
        postprocess = self.postprocess(
            output[
                self.indexing_plugin.join_indices(
                    dimension_indices=self.location.dimension_index,
                )
            ]
        )
        self.storage_plugin.add_payload(location=self.location, payload=postprocess)

    def setup(self, module: "torch.nn.Module") -> "torch.nn.Module":
        self.handle = module.register_forward_hook(self.output_monitor_hook)

        if self.move_to_first:
            # we push the current hook to the beginning of the queue,
            # as this is
            # for a monitor and its deployment must be before
            # the fault injection
            # we use move_to_end with last=False to move it to the beginning
            # of the OrderedDict
            # mypy has issues with Optional being set before, as it does not check them
            # sometimes the following 2 lines fail, use type: ignore[union-attr]
            # for both
            self.handle.hooks_dict_ref().move_to_end(
                self.handle.id,
                last=False,
            )

        return module
