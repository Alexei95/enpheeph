import pathlib
import sys

import pytorch_lightning
import pytorch_lightning.utilities.cli
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().\
    parent.parent.parent.parent.parent.resolve()
CONFIG_ROOT = PROJECT_ROOT / 'config'
DEFAULT_CONFIGURATION_FILE = CONFIG_ROOT / 'custom_config.yml'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.ml.pl.models.plwrapperlightningcli


class LightningCLI(pytorch_lightning.utilities.cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        # LINKING CANNOT BE USED IF USING SUBCLASSES
        # to link together dataloader batch size and model batch size
        parser.link_arguments('data.batch_size', 'model.batch_size')
        # to link together the number of classes in the dataset
        # NOT NEEDED FOR NOW
        # parser.link_arguments(
        #         'data.num_classes',
        #         'model.num_classes',
        #         apply_on='instantiate'
        # )
        # actually these ones are not needed, as we can simply use the
        # class_path interface for the classes with instantiate_class
        # they are useful only for providing extra support for optimizer
        # arguments, so without going through the docs, which is not something
        # important for now
        # to add all the arguments of possible optimizer in the interface
        # parser.add_optimizer_args(
        #         tuple(torch.optim.Optimizer.__subclasses__()),
        #         link_to="model.optimizer_classes",
        # )
        # parser.add_lr_scheduler_args(
        #         (
        #                 tuple(
        #                         torch.optim.lr_scheduler.
        #                         _LRScheduler.__subclasses__()
        #                 ) + (torch.optim.lr_scheduler.ReduceLROnPlateau, )
        #         ),
        #         link_to="model.lr_scheduler_classes",
        # )

    def before_fit(self):
        self.trainer.tune(self.model)

    def after_fit(self):
        self.trainer.test(**self.fit_kwargs)


# we create the object and append the project path to sys.path only if this
# file is being run as a script
if __name__ == '__main__':
    cli = LightningCLI(
            model_class=(
                    src.ml.pl.models.plwrapperlightningcli.
                    PLWrapperLightningCLI
            ),
            datamodule_class=pytorch_lightning.LightningDataModule,
            # to allow subclasses to be used for the model and for the
            # datamodule
            # subclass_mode_model=True,
            subclass_mode_data=True,
            # we add this to overwrite the configuration
            # it is useful for debug purposes
            save_config_overwrite=True
            # we pass a default config using the custom one, which can be
            # overriden by the --config flag
            # however for the way the CLI works, this config is overriden
            # by the --config flag completely, without intersection
            # parser_kwargs={
            #         'default_config_files': [
            #                 str(DEFAULT_CONFIGURATION_FILE)
            #         ],
            # },
)
