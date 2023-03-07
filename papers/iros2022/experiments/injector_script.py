# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import collections.abc
import datetime
import functools
import importlib
import operator
import os
import pathlib
import random
import sys
import typing

import flash
import pytorch_lightning
import torch
import torch.quantization
import torchinfo

import enpheeph.injections.fpquantizedoutputpytorchfault
import enpheeph.injections.monitorabc

# for pickle to avoid explosion
if str(pathlib.Path(__file__).parent / "results/configs/snn_training") not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parent / "results/configs/snn_training"))

    sys.path.pop()


CURRENT_DIR = pathlib.Path(__file__).absolute().parent
RESULTS_DIRECTORY = CURRENT_DIR / "results"
DATASET_DIRECTORY = pathlib.Path("/shared/ml/datasets/vision/")


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


def compute_layer_module_name(
    layer: torchinfo.layer_info.LayerInfo,
) -> str:
    # with this while loop we compute the layer name from the layer itself
    # we simply join all the parent variable names until we reach the main model
    module_name = layer.var_name
    p = layer.parent_info
    # we need to skip the main model as it would add an extra dot
    # we can find it as its depth is 0
    while p is not None and p.depth > 0:
        module_name = p.var_name + "." + module_name
        p = p.parent_info
    return module_name


# we can create the injections
def create_injections_for_layer_with_randomness_value(
    config: typing.Dict[str, typing.Any],
    layer: torchinfo.layer_info.LayerInfo,
    randomness: float,
) -> typing.Generator[enpheeph.utils.data_classes.InjectionLocationABC, None, None]:
    module_name = compute_layer_module_name(layer=layer)

    # we check if the layer is ok to run a fault injection on
    if not layer.is_leaf_layer or not layer.executed:
        return []

    print(f"Layer: {module_name}\nRandomness: {randomness}\n\n")

    injections = []

    # inj_type = "activation"
    # inj_type = "quantized_activation"
    inj_type = "sparse_activation"
    # inj_type = "weight"

    if inj_type == "activation":
        # we multiply by a very small number > 1 to increase the range and cover also 1
        # we skip the batch size as the first dimension
        shape = layer.output_size[1:]
        if (
            config.get("injection_config", {}).get(
                "indexing_dimension_dict",
                enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
            )
            != enpheeph.utils.constants.PYTORCH_DIMENSION_DICT
        ):
            # we remove the extra time dimension if it is an SNN
            shape = shape[1:]
        mask = torch.rand(*shape, device="cpu") * 1.00000001 <= randomness
        inj = enpheeph.injections.OutputPyTorchFault(
            location=enpheeph.utils.data_classes.FaultLocation(
                module_name=module_name,
                parameter_type=enpheeph.utils.enums.ParameterType.Activation,
                dimension_index={
                    enpheeph.utils.enums.DimensionType.Batch: ...,
                    enpheeph.utils.enums.DimensionType.Time: ...,
                },
                dimension_mask={
                    enpheeph.utils.enums.DimensionType.Tensor: mask.tolist(),
                },
                bit_index=random.sample(
                    list(range(config.get("injection_config", {}).get("bitwidth", 32))),
                    1,
                ),
                bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
            ),
            low_level_torch_plugin=enpheeph.injections.plugins.mask.autopytorchmaskplugin.AutoPyTorchMaskPlugin(),
            indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
                dimension_dict=config.get("injection_config", {}).get(
                    "indexing_dimension_dict",
                    enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
                ),
            ),
        )
    elif inj_type == "sparse_activation":
        shape = layer.output_size[1:]
        approx_n_elements = functools.reduce(operator.mul, shape)
        inj = enpheeph.injections.DenseSparseOutputPyTorchFault(
            location=enpheeph.utils.data_classes.FaultLocation(
                module_name=module_name,
                parameter_type=enpheeph.utils.enums.ParameterType.Activation
                | enpheeph.utils.enums.ParameterType.Sparse
                | enpheeph.utils.enums.ParameterType.Value,
                dimension_index={
                    enpheeph.utils.enums.DimensionType.Tensor: random.sample(
                        list(range(approx_n_elements)),
                        abs(int((random.random() - randomness) * approx_n_elements)),
                    ),
                },
                dimension_mask=None,
                bit_index=random.sample(
                    list(range(config.get("injection_config", {}).get("bitwidth", 32))),
                    1,
                ),
                bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
            ),
            low_level_torch_plugin=enpheeph.injections.plugins.mask.autopytorchmaskplugin.AutoPyTorchMaskPlugin(),
            indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
                dimension_dict=config.get("injection_config", {}).get(
                    "indexing_dimension_dict",
                    enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
                ),
            ),
        )
    elif inj_type == "quantized_activation":
        # we multiply by a very small number > 1 to increase the range and cover also 1
        # we skip the batch size as the first dimension
        shape = layer.output_size[1:]
        if (
            config.get("injection_config", {}).get(
                "indexing_dimension_dict",
                enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
            )
            != enpheeph.utils.constants.PYTORCH_DIMENSION_DICT
        ):
            # we remove the extra time dimension if it is an SNN
            shape = shape[1:]
        mask = torch.rand(*shape, device="cpu") * 1.00000001 <= randomness
        inj = enpheeph.injections.fpquantizedoutputpytorchfault.FPQuantizedOutputPyTorchFault(
            location=enpheeph.utils.data_classes.FaultLocation(
                module_name=module_name,
                parameter_type=enpheeph.utils.enums.ParameterType.Activation,
                dimension_index={
                    enpheeph.utils.enums.DimensionType.Batch: ...,
                    enpheeph.utils.enums.DimensionType.Time: ...,
                },
                dimension_mask={
                    enpheeph.utils.enums.DimensionType.Tensor: mask.tolist(),
                },
                bit_index=random.sample(
                    list(range(config.get("injection_config", {}).get("bitwidth", 32))),
                    1,
                ),
                bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
            ),
            low_level_torch_plugin=enpheeph.injections.plugins.mask.autopytorchmaskplugin.AutoPyTorchMaskPlugin(),
            indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
                dimension_dict=config.get("injection_config", {}).get(
                    "indexing_dimension_dict",
                    enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
                ),
            ),
        )
    elif inj_type == "weight":
        # we multiply by a very small number > 1 to increase the range and cover also 1
        # we skip the batch size as the first dimension
        mask = (
            torch.rand(*layer.module.weight.shape, device="cpu") * 1.00000001
            <= randomness
        )
        inj = enpheeph.injections.WeightPyTorchFault(
            location=enpheeph.utils.data_classes.FaultLocation(
                module_name=module_name,
                parameter_type=enpheeph.utils.enums.ParameterType.Weight,
                parameter_name="weight",
                dimension_index={
                    enpheeph.utils.enums.DimensionType.Batch: ...,
                    enpheeph.utils.enums.DimensionType.Time: ...,
                },
                dimension_mask={
                    enpheeph.utils.enums.DimensionType.Tensor: mask.tolist(),
                },
                bit_index=random.sample(
                    list(range(config.get("injection_config", {}).get("bitwidth", 32))),
                    1,
                ),
                bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
            ),
            low_level_torch_plugin=enpheeph.injections.plugins.mask.autopytorchmaskplugin.AutoPyTorchMaskPlugin(),
            indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
                dimension_dict=config.get("injection_config", {}).get(
                    "indexing_dimension_dict",
                    enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
                ),
            ),
        )

    injections.append(inj)

    return injections


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
    parser.add_argument(
        "--csv-results",
        type=pathlib.Path,
        default=pathlib.Path(os.devnull),
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

    injection_type_group = parser.add_mutually_exclusive_group()
    injection_type_group.add_argument(
        "--random",
        action="store_true",
    )
    injection_type_group.add_argument(
        "--custom",
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
    else:
        device_config = {}
    if namespace.random:
        random_config = importlib.import_module("random_multi_config")
        injection_config = random_config.config()
    else:
        injection_config = {}
    # we remove the previously appended path to leave it as is
    sys.path.pop()

    # we instantiate the config from the imported module
    initial_config = config_module.config(
        dataset_directory=DATASET_DIRECTORY,
        model_weight_file=namespace.model_weight_file,
        storage_file=namespace.storage_file,
    )
    config = recursive_dict_update(initial_config, device_config)
    config = recursive_dict_update(initial_config, injection_config)
    config = safe_recursive_instantiate_dict(config)

    pytorch_lightning.seed_everything(**config.get("seed_everything", {}))

    trainer = config["trainer"]
    model = config["model"]
    # model = config["model_post_init"](model)
    datamodule = config["datamodule"]

    # if the static quantization was selected
    # we train the model for an additional epoch (set in the default trainer config)
    # to be able to create the proper static quantization weights + activations
    # **NOTE**: static quantization is not supported on GPU
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
    # **NOTE**: dynamic quantization does not work on GPU
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
    time = datetime.datetime.utcnow()
    res = trainer.test(
        model,
        dataloaders=datamodule.test_dataloader(),
    )[
        0
    ]  # we have only one test dataloader
    execution_time = datetime.datetime.utcnow() - time
    namespace.csv_results.parent.mkdir(parents=True, exist_ok=True)
    with namespace.csv_results.open("a") as f:
        f.write(
            f"randomness,layer_name,execution_time,{','.join(str(x) for x in res.keys())}\n"
        )
        f.write(
            f"0,-,{str(execution_time.total_seconds())},{','.join(str(x) for x in res.values())}\n"
        )

    if config.get("injection_config", {}).get("custom", True):
        # we do only monitors if we activate any injection
        if config["injection_handler"].activate(
            [
                monitor
                for monitor in config["injection_handler"].injections
                if isinstance(monitor, enpheeph.injections.monitorabc.MonitorABC)
            ]
        ):
            print("\n\nOnly monitors\n\n")
            time = datetime.datetime.utcnow()
            res = trainer.test(
                model,
                dataloaders=datamodule.test_dataloader(),
            )[
                0
            ]  # we have only one test dataloader
            execution_time = datetime.datetime.utcnow() - time
            with namespace.csv_results.open("a") as f:
                f.write(
                    f"0,-,{str(execution_time.total_seconds())},{','.join(str(x) for x in res.values())}\n"
                )

        print("\n\nAll injections\n\n")
        config["injection_handler"].activate()
        time = datetime.datetime.utcnow()
        res = trainer.test(
            model,
            dataloaders=datamodule.test_dataloader(),
        )[
            0
        ]  # we have only one test dataloader
        execution_time = datetime.datetime.utcnow() - time
        with namespace.csv_results.open("a") as f:
            f.write(
                f"custom,custom,{str(execution_time.total_seconds())},{','.join(str(x) for x in res.values())}\n"
            )
    else:
        inp = next(iter(datamodule.test_dataloader()))
        if isinstance(inp, dict):
            inp = inp[flash.core.data.data_source.DefaultDataKeys.INPUT]
            shape = list(inp.shape)
        else:
            inp = inp[0]
            shape = list(inp.shape)
            shape[1] = 1
        # otherwise it does not work for SNNs
        shape[0] = 1
        # we take the shape from the datamodule
        summary = torchinfo.summary(model=model, input_size=shape, device="cpu")
        #
        allowed_layers = config.get("injection_config", {}).get("layers", None)
        for r in config.get("injection_config", {}).get("randomness", []):
            for layer in summary.summary_list:
                if (
                    allowed_layers is not None
                    and compute_layer_module_name(layer) not in allowed_layers
                ):
                    continue
                config["injection_handler"].remove_injections()
                injections = create_injections_for_layer_with_randomness_value(
                    config=config, layer=layer, randomness=r
                )
                config["injection_handler"].add_injections(injections)
                config["injection_handler"].deactivate()

                # we do only monitors if we activate any injection
                if config["injection_handler"].activate(
                    [
                        monitor
                        for monitor in config["injection_handler"].injections
                        if isinstance(
                            monitor, enpheeph.injections.monitorabc.MonitorABC
                        )
                    ]
                ):
                    print("\n\nOnly monitors\n\n")
                    time = datetime.datetime.utcnow()
                    res = trainer.test(
                        model,
                        dataloaders=datamodule.test_dataloader(),
                    )[
                        0
                    ]  # we have only one test dataloader
                    execution_time = datetime.datetime.utcnow() - time
                    with namespace.csv_results.open("a") as f:
                        f.write(
                            f"0,-,{str(execution_time.total_seconds())},{','.join(str(x) for x in res.values())}\n"
                        )

                if config["injection_handler"].activate():
                    print("\n\nAll injections\n\n")
                    time = datetime.datetime.utcnow()
                    res = trainer.test(
                        model,
                        dataloaders=datamodule.test_dataloader(),
                    )[
                        0
                    ]  # we have only one test dataloader
                    execution_time = datetime.datetime.utcnow() - time
                    with namespace.csv_results.open("a") as f:
                        f.write(
                            f"{str(r)},{compute_layer_module_name(layer)},{str(execution_time.total_seconds())},{','.join(str(x) for x in res.values())}\n"
                        )

    print("\n\nAgain no injections at all\n\n")
    config["injection_handler"].deactivate()
    time = datetime.datetime.utcnow()
    res = trainer.test(
        model,
        dataloaders=datamodule.test_dataloader(),
    )[
        0
    ]  # we have only one test dataloader
    execution_time = datetime.datetime.utcnow() - time
    with namespace.csv_results.open("a") as f:
        f.write(
            f"0,-,{str(execution_time.total_seconds())},{','.join(str(x) for x in res.values())}\n"
        )


if __name__ == "__main__":
    main()
