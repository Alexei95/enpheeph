import pytorch_lightning
import pytorch_lightning.utilities.cli


cli = pytorch_lightning.utilities.cli.LightningCLI(
        model_class=pytorch_lightning.LightningModule,
        datamodule_class=pytorch_lightning.LightningDataModule,
        # to allow subclasses to be used for the model and for the datamodule
        subclass_mode_model=True,
        subclass_mode_data=True,
)
