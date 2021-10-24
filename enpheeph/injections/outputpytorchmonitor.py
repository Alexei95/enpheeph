# -*- coding: utf-8 -*-
import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmonitorpostprocessormixin
import enpheeph.injections.plugins.storagepluginabc
import enpheeph.utils.data_classes
import enpheeph.utils.enums


class OutputPyTorchMonitor(
    enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
    (
        enpheeph.injections.mixins.pytorchmonitorpostprocessormixin.PyTorchMonitorPostProcessorMixIn
    ),
):
    def __init__(
        self,
        monitor_location: enpheeph.utils.data_classes.InjectionLocation,
        enabled_metrics: enpheeph.utils.enums.MonitorMetric,
        storage_plugin: enpheeph.injections.plugins.storagepluginabc.StoragePluginABC,
        move_to_first: bool = True,
    ):
        super().__init__()

        self.monitor_location = monitor_location
        self.enabled_metrics = enabled_metrics
        self.storage_plugin = storage_plugin
        self.move_to_first = move_to_first

        self.handle = None

    @property
    def module_name(self):
        return self.monitor_location.module_name

    def output_monitor_hook(self, module, input, output):
        # NOTE: no support for bit_index yet
        postprocess = self.postprocess(output[self.monitor_location.tensor_index])
        self.storage_plugin.add_dict(postprocess)
        self.storage_plugin.submit_eol()

    def setup(self, module):
        self.handle = module.register_forward_hook(self.output_monitor_hook)

        if self.move_to_first:
            # we push the current hook to the beginning of the queue,
            # as this is
            # for a monitor and its deployment must be before
            # the fault injection
            # we use move_to_end with last=False to move it to the beginning
            # of the OrderedDict
            self.handle.hooks_dict_ref().move_to_end(
                self.handle.id, last=False,
            )

        return module

    def teardown(self, module):
        self.handle.remove()

        self.handle = None

        return module
