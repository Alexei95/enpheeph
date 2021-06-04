import copy
import json
import pathlib
import pprint
import shutil
import sys

import pytorch_lightning

PROJECT_ROOT = pathlib.Path(__file__).resolve().\
    parent.parent.parent.parent.parent.resolve()
CONFIG_ROOT = PROJECT_ROOT / 'config'
DEFAULT_CONFIGURATION_FILE = CONFIG_ROOT / 'custom_config.yml'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.cli.utils.argumentparser
import src.dnn.pl.utils.callbacks.saveconfigcallback
import src.utils.json.jsonparser
import src.utils.json.handlers.callablehandler
import src.utils.json.handlers.objecthandler


class PLTrainerCLI(
        src.utils.json.jsonparser.JSONParser,
):
    # default subdirectory for saving configs
    DEFAULT_SUBDIRECTORY = 'configs'
    # default keys and values for the default configs
    DEFAULT_CONFIG_KEY = 'config'
    DEFAULT_CONFIG_VALUE = []
    DEFAULT_SAVE_CONFIG_KEY = 'save_config'
    DEFAULT_SAVE_CONFIG_FLAG = True
    DEFAULT_SAVE_CONFIG_VALUE = pathlib.Path('.')
    DEFAULT_TRAINER_ROOT_DIR_KEY = 'default_root_dir'
    DEFAULT_CALLBACKS_KEY = 'callbacks'
    DEFAULT_CALLBACKS_VALUE = []
    # keys to be used in the config
    CONFIG_KEY = 'config'
    TRAINER_KEY = 'trainer'
    TRAINER_DEFAULT_VALUE = {}
    MODEL_KEY = 'model'
    DATAMODULE_KEY = 'datamodule'
    TRAINER_CLASS_KEY = 'trainer_class'
    DATAMODULE_CLASS_KEY = 'datamodule_class'
    MODEL_CLASS_KEY = 'model_class'
    # key and value for Trainer.tune config
    TRAINER_TUNE_KEY = 'trainer_tune'
    TRAINER_TUNE_DEFAULT = False

    def __init__(self, defaults=None):
        super().__init__()

        self.CONFIGS_TO_CHECK = [
                {'key': key, 'class_key': class_key}
                for key, class_key in zip(
                        (
                                self.TRAINER_KEY,
                                self.MODEL_KEY,
                                self.DATAMODULE_KEY
                        ),
                        (
                                self.TRAINER_CLASS_KEY,
                                self.MODEL_CLASS_KEY,
                                self.DATAMODULE_CLASS_KEY
                        )
                )
        ]

        self.defaults = (
                {
                        self.DEFAULT_CONFIG_KEY: self.DEFAULT_CONFIG_VALUE,
                        self.DEFAULT_SAVE_CONFIG_KEY:
                        self.DEFAULT_SAVE_CONFIG_VALUE,
                }
                if defaults is None
                else defaults
        )

        self.DecoderDispatcher.register(
                src.utils.json.handlers.callablehandler.\
                CallableHandler.CALLABLE_HANDLER_DEFAULT_STRING,
                src.utils.json.handlers.callablehandler.\
                CallableHandler.decode_json
        )
        self.DecoderDispatcher.register(
                src.utils.json.handlers.objecthandler.\
                ObjectHandler.OBJECT_HANDLER_DEFAULT_STRING,
                src.utils.json.handlers.objecthandler.\
                ObjectHandler.decode_json
        )

        self.parser = src.cli.utils.argumentparser.ArgumentParser()
        self.namespace = None
        self.config = None
        self.raw_config = None
        self.json_raw_config = None
        self.config_paths = None
        self.trainer = None
        self.model = None
        self.datamodule = None

    def run(self):
        self.add_default_arguments(parser=self.parser, defaults=self.defaults)
        self.add_extra_arguments(parser=self.parser, defaults=self.defaults)
        self.parse_arguments(parser=self.parser)
        self.postprocess_parsed_arguments(namespace=self.namespace)
        self.load_raw_configs(namespace=self.namespace)
        self.postprocess_loaded_raw_configs(config=self.raw_config)
        self.convert_raw_config_to_json_config(config=self.raw_config)
        self.load_config(config=self.json_raw_config)
        self.postprocess_loaded_configs(config=self.config)
        self.check_config(config=self.config)
        self.init_trainer(config=self.config)
        self.init_model(config=self.config)
        self.init_datamodule(config=self.config)
        self.before_tune(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.tune(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.after_tune(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.before_fit(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.fit(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.after_fit(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.before_test(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.test(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )
        self.after_test(
                trainer=self.trainer,
                model=self.mode,
                datamodule=self.datamodule,
                config=self.config
        )

    def add_default_arguments(self, parser, defaults):
        # when using '+', if the option starts with -- it is considered
        # optional, but it must have at least one argument when used
        # so "--config a" or "" are ok but "--config" is not
        # hence we add required=True
        parser.add_argument(
                '-c',
                '--config',
                nargs='+',
                required=True,
                type=pathlib.Path,
                default=defaults.get(
                        self.DEFAULT_CONFIG_KEY,
                        self.DEFAULT_CONFIG_VALUE
                ),
                help='to provide the paths for the JSON configs to be loaded',
        )
        # we use the same flag so that if we get the flag we can use the
        # default path if no Trainer path is provided
        parser.add_argument(
                '--save-config',
                action='store',
                nargs='?',
                default=defaults.get(
                        self.DEFAULT_SAVE_CONFIG_KEY,
                        copy.deepcopy(self.DEFAULT_SAVE_CONFIG_FLAG),
                ),
                const=defaults.get(
                        self.DEFAULT_SAVE_CONFIG_KEY,
                        copy.deepcopy(self.DEFAULT_SAVE_CONFIG_FLAG),
                ),
                type=pathlib.Path,
                help=(
                        'if enabled it will save the configuration in '
                        'the Trainer default_root_dir'
                ),
        )

    def add_extra_arguments(self, parser, defaults):
        pass

    def parse_arguments(self, parser, args=None, namespace=None):
        self.namespace = parser.parse_args(args=args, namespace=namespace)

    def postprocess_parsed_arguments(self, namespace):
        pass

    def load_raw_configs(self, namespace):
        config_paths = getattr(namespace, self.DEFAULT_CONFIG_KEY)

        # we load the JSON config, without decoding
        config = self.load_paths(config_paths, raw=True)

        # here we check the save config path
        # if it is enabled we attach it in the configuration
        # for the path, if not given we get the Trainer default_root_dir
        # path
        save_config_value = getattr(namespace, self.DEFAULT_SAVE_CONFIG_KEY)
        if save_config_value:
            if isinstance(
                    save_config_value,
                    pathlib.Path
            ):
                save_path = pathlib.Path(save_config_value).resolve()
            else:
                save_path = (
                        pathlib.Path(
                                config.get(
                                    self.TRAINER_KEY,
                                    copy.deepcopy(self.TRAINER_DEFAULT_VALUE),
                                ).get(
                                        '__kwargs__', {}
                                ).get(
                                        self.DEFAULT_TRAINER_ROOT_DIR_KEY,
                                        copy.deepcopy(
                                                self.DEFAULT_SAVE_CONFIG_VALUE
                                        ),
                                )
                        ).resolve() / self.DEFAULT_SUBDIRECTORY
                ).resolve()

            # we generate the correct JSON for SaveConfigCallback
            complete_dest_dir = save_path.resolve() / self.DEFAULT_SUBDIRECTORY
            complete_dest_dir.mkdir(exist_ok=True, parents=True)
            # we cycle through the configurations
            for index, c in enumerate(config_paths):
                # we resolve the path
                c = c.resolve()
                # we get the complete suffix
                config_suffix = ''.join(c.suffixes)
                # we generate a new name using the index followed by the suffix
                config_name = c.with_suffix(
                        f".{index}{config_suffix}"
                ).name

                shutil.copy2(c, complete_dest_dir / config_name)

        self.raw_config = config
        self.config_paths = config_paths

    def _save_configs(self, ):
        pass

    def postprocess_loaded_raw_configs(self, config):
        pass

    def convert_raw_config_to_json_config(self, config):
        self.json_raw_config = json.dumps(config)

    def load_config(self, config):
        self.config = self.load_strings([config])

    def postprocess_loaded_configs(self, config):
        pass

    # we use this function to check the behaviour of all the config, before
    # starting to load all the objects
    def check_config(self, config):
        for dict_ in self.CONFIGS_TO_CHECK:
            # we check whether trainer is in the config
            try:
                instance = config[dict_['key']]
            # if not we raise an error
            except KeyError:
                raise ValueError(
                        "config does not "
                        "contain required key '{}'".format(dict_['key'])
                )

            # if the value is not a Trainer instance, we check whether there
            # is a __class__ key in the config
            if not isinstance(instance, pytorch_lightning.Trainer):
                try:
                    config[dict_['class_key']]
                except KeyError:
                    raise ValueError(
                        "config does not "
                        "contain required class key '{1}'".format(
                                dict_['class_key'],
                        )
                    )

    def init_trainer(self, config):
        # if the value is not a Trainer instance
        # we build it from the trainer_class config
        if not isinstance(config[self.TRAINER_KEY], pytorch_lightning.Trainer):
            trainer = config[self.TRAINER_CLASS_KEY](
                    **config[self.TRAINER_KEY]
            )
        else:
            trainer = config[self.TRAINER_KEY]

        self.trainer = trainer

    def init_model(self, config):
        # if the value is not a LightningModule instance
        # we build it from the model_class config
        if not isinstance(
                config[self.MODEL_KEY],
                pytorch_lightning.LightningModule,
        ):
            model = config[self.MODEL_CLASS_KEY](
                    **config[self.MODEL_KEY]
            )
        else:
            model = config[self.MODEL_KEY]

        self.model = model

    def init_datamodule(self, config):
        # if the value is not a LightningDataModule instance
        # we build it from the datamodule_class config
        if not isinstance(
                config[self.DATAMODULE_KEY],
                pytorch_lightning.LightningDataModule,
        ):
            datamodule = config[self.DATAMODULE_CLASS_KEY](
                    **config[self.DATAMODULE_KEY]
            )
        else:
            datamodule = config[self.DATAMODULE_KEY]

        self.datamodule = datamodule

    def before_tune(self, trainer, model, datamodule, config):
        pass

    def tune(self, trainer, model, datamodule, config):
        # we check the tune flag/dict, if it is False
        # we skip the tuning
        # here we consider True an empty dict, as we check using "is"
        if config.get(
                self.TRAINER_TUNE_KEY,
                copy.deepcopy(self.TRAINER_TUNE_DEFAULT),
        ) is False:
            return

        kwargs = config.get(self.TRAINER_TUNE_KEY, {})
        trainer.tune(
                model=model,
                datamodule=datamodule,
                **kwargs,
        )

    def after_tune(self, trainer, model, datamodule, config):
        pass

    def before_fit(self, trainer, model, datamodule, config):
        pass

    def fit(self, trainer, model, datamodule, config):
        trainer.fit(model=model, datamodule=datamodule)

    def after_fit(self, trainer, model, datamodule, config):
        pass

    def before_test(self, trainer, model, datamodule, config):
        pass

    def test(self, trainer, model, datamodule, config):
        trainer.test(model=model, datamodule=datamodule)

    def after_test(self, trainer, model, datamodule, config):
        pass

    def before_predict(self, trainer, model, datamodule, config):
        pass

    def predict(self, trainer, model, datamodule, config):
        trainer.test(model=model, datamodule=datamodule)

    def after_predict(self, trainer, model, datamodule, config):
        pass


# we create the object only if this
# file is being run as a script
if __name__ == '__main__':
    cli = PLTrainerCLI()
    cli.run()
