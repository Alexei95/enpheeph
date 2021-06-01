import pathlib
import sys

import pytorch_lightning
import pytorch_lightning.utilities.cli

PROJECT_ROOT = pathlib.Path(__file__).resolve().\
    parent.parent.parent.parent.parent.resolve()
CONFIG_ROOT = PROJECT_ROOT / 'config'
DEFAULT_CONFIGURATION_FILE = CONFIG_ROOT / 'custom_config.yml'


class PLTrainerCLI(pytorch_lightning.utilities.cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        pass

    def parse_arguments(self):
        self.config = self.parser.parse_args(_skip_check=True)


# we create the object and append the project path to sys.path only if this
# file is being run as a script
if __name__ == '__main__':
    # we need to append the project root to be able to import the custom
    # wrapper classes inside the project, which otherwise would be unreachable
    # since this file is called as a script
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    cli = PLTrainerCLI(
            model_class=pytorch_lightning.LightningModule,
            datamodule_class=pytorch_lightning.LightningDataModule,
            # to allow subclasses to be used for the model and for the
            # datamodule
            subclass_mode_model=True,
            subclass_mode_data=True,
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
