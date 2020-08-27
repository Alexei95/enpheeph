import pytorch_lightning as pl

from . import callbackinterfaceabc
from ..modelwrappers import pytorchmodelwrapper


class TestLightningCallbackInterface(pl.Callback, callbackinterfaceabc.CallbackInterfaceABC):
    def __init__(self, wrapper_cls=pytorchmodelwrapper.PyTorchModelWrapper, *args, **kwargs):
        kwargs.update({'wrapper_cls': wrapper_cls})
        super().__init__(*args, **kwargs)

    def on_test_start(self, trainer, pl_module):
        self.setup_fi(model=pl_module)

    def on_test_end(self, trainer, pl_module):
        self.restore_fi(model=pl_module)


CALLBACK = {TestLightningCallbackInterface.__name__: TestLightningCallbackInterface}
