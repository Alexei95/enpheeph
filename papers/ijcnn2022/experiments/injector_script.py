# -*- coding: utf-8 -*-
import argparse
import collections.abc
import importlib
import pathlib
import sys
import typing

import pytorch_lightning
import torch
import torch.quantization

import enpheeph.injections.monitorabc


CURRENT_DIR = pathlib.Path(__file__).absolute().parent
RESULTS_DIRECTORY = CURRENT_DIR / "results"
CIFAR10_DIRECTORY = pathlib.Path("/shared/ml/datasets/vision/CIFAR10/")


# it overwrites the keys with the new value
def recursive_dict_update(original: typing.Dict, mergee: typing.Dict) -> typing.Dict:
    for k, v in mergee.items():
        if k in original and isinstance(original[k], collections.abc.Mapping):
            original[k] = recursive_dict_update(original[k], v)
        else:
            original[k] = v
    return original


def safe_recursive_instantiate_dict(config: typing.Any) -> typing.Any:
    # if we have a mapping-like, e.g. dict, we check whether it must be directly
    # instantiated
    # if yes we return the final object, otherwise we call the function on each
    # value in the dict
    if isinstance(config, collections.abc.Mapping):
        # we use issubset to allow extra values if needed for other purposes
        # which are not used in this instantiation and will be lost
        if set(config.keys()) == {"callable", "callable_args"}:
            # we need to pass the instantiated version of the config dict
            return config["callable"](
                **safe_recursive_instantiate_dict(config["callable_args"])
            )
        # otherwise we create a copy and we instantiate each value with the
        # corresponding key
        # copy.deepcopy does not work, we skip it
        new_config = config
        for key, value in config.items():
            new_config[key] = safe_recursive_instantiate_dict(value)
        return new_config
    # if we have a sequence-like, e.g. list, we create the same class
    # where each element is instantiated
    elif isinstance(config, (list, tuple, set)):
        new_config = config.__class__(
            [safe_recursive_instantiate_dict(v) for v in config]
        )
        return new_config
    # if we have a generic element, e.g. str, we return it as-is
    else:
        return config


def setup_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--model-weight-file",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--storage-file",
        type=pathlib.Path,
        required=True,
    )
    mutex_quantize_group = parser.add_mutually_exclusive_group()
    mutex_quantize_group.add_argument(
        "--static-quantize",
        action="store_true",
    )
    mutex_quantize_group.add_argument(
        "--dynamic-quantize",
        action="store_true",
    )
    mutex_device_group = parser.add_mutually_exclusive_group()
    mutex_device_group.add_argument(
        "--cpu",
        action="store_true",
    )
    mutex_device_group.add_argument(
        "--gpu",
        action="store_true",
    )

    return parser


def main(args=None):

    parser = setup_argument_parser()

    namespace = parser.parse_args(args=args)

    # here we append the path of the configuration to sys.path so that it can
    # be easily imported
    sys.path.append(str(namespace.config.parent))
    # we import the module by taking its name
    config_module = importlib.import_module(namespace.config.with_suffix("").name)
    # we select the devices on which we run the simulation
    if namespace.gpu:
        gpu_config = importlib.import_module("gpu_config")
        device_config = gpu_config.config()
    elif namespace.cpu:
        cpu_config = importlib.import_module("cpu_config")
        device_config = cpu_config.config()
    # we remove the previously appended path to leave it as is
    sys.path.pop()

    # we instantiate the config from the imported module
    initial_config = config_module.config(
        dataset_directory=CIFAR10_DIRECTORY,
        model_weight_file=namespace.model_weight_file,
        storage_file=namespace.storage_file,
    )
    config = recursive_dict_update(initial_config, device_config)
    config = safe_recursive_instantiate_dict(initial_config)

    pytorch_lightning.seed_everything(**config.get("seed_everything", {}))

    trainer = config["trainer"]
    model = config["model"]
    # model = config["model_post_init"](model)
    datamodule = config["datamodule"]

    # if the static quantization was selected
    # we train the model for an additional epoch (set in the default trainer config)
    # to be able to create the proper static quantization weights + activations
    if namespace.static_quantize:
        config["injection_handler"].deactivate()
        trainer.callbacks.append(
            pytorch_lightning.callbacks.QuantizationAwareTraining()
        )

        trainer.fit(
            model,
            datamodule=datamodule,
        )
    # with the dynamic quantization we quantize only the weights by a fixed
    # configuration
    # dynamic quantization does not work on GPU, due to PyTorch missing the kernels
    elif namespace.dynamic_quantize:
        model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec=config.get("dynamic_quantization_config", {}).get(
                "qconfig",
                {
                    torch.nn.Linear,
                    torch.nn.LSTM,
                    torch.nn.GRU,
                    torch.nn.LSTMCell,
                    torch.nn.RNNCell,
                    torch.nn.GRUCell,
                    torch.nn.EmbeddingBag,
                },
            ),
            dtype=config.get("dynamic_quantization_config", {}).get(
                "qdtype",
                torch.qint8,
            ),
            # we need to force in-place otherwise Flash Models cannot be deep-copied
            inplace=True,
        )

    print("\n\nNo injections at all\n\n")
    config["injection_handler"].deactivate()
    trainer.test(
        model,
        dataloaders=datamodule.test_dataloader(),
    )

    print("\n\nOnly monitors\n\n")
    config["injection_handler"].activate(
        [
            monitor
            for monitor in config["injection_handler"].injections
            if isinstance(monitor, enpheeph.injections.monitorabc.MonitorABC)
        ]
    )
    trainer.test(
        model,
        dataloaders=datamodule.test_dataloader(),
    )

    print("\n\nAll injections\n\n")
    config["injection_handler"].activate()
    trainer.test(
        model,
        dataloaders=datamodule.test_dataloader(),
    )

    print("\n\nAgain no injections at all\n\n")
    config["injection_handler"].deactivate()
    trainer.test(
        model,
        dataloaders=datamodule.test_dataloader(),
    )


if __name__ == "__main__":
    main()
