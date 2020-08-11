import pytorch_lightning as pl


class TestLightningCallbackInterface(pl.Callback):
    def __init__(self, fault_injector):
        self._fault_injector = fault_injector

    def on_test_start(self, trainer, pl_module):
        self._fault_injector.setup_fi(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._fault_injector.restore_fi(pl_module)
