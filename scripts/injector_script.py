# -*- coding: utf-8 -*-
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
import datetime
import importlib
import pathlib
import sys
import typing
import flash

import pytorch_lightning

import enpheeph
import enpheeph.helpers.importancesampling
import enpheeph.injections.abc
import enpheeph.injections.pruneddensetosparseactivationpytorchfault


def get_generic_injection_callback(result_database) -> (
    pytorch_lightning.Callback
):
    result_database = result_database.absolute()
    result_database.parent.mkdir(exist_ok=True, parents=True)

    pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()
    storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
        db_url="sqlite:///" + str(result_database)
    )

    injection_handler = enpheeph.handlers.InjectionHandler(
        injections=[],
        library_handler_plugin=pytorch_handler_plugin,
    )

    # we delay the instantiation of the callback to allow the saving of the
    # current configuration
    callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
        injection_handler=injection_handler,
        storage_plugin=storage_plugin,
        # this config used to contain the complete system config: trainer + model +
        # dataset, including the configuration for injections
        # extra_session_info=config,
    )
    return callback


def get_trainer_config(args=sys.argv) -> typing.Dict[str, typing.Any]:
    config = pathlib.Path(args[1]).absolute()

    sys.path.append(str(config.parent))

    module_name = config.with_suffix("").name

    module = importlib.import_module(module_name)

    sys.path.pop()

    config_dict = module.config()

    return config_dict


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--python-config", action="store", type=pathlib.Path, required=True)
    parser.add_argument("--target-csv", action="store", type=pathlib.Path, required=False, default="results/table.csv")
    parser.add_argument("--seed", action="store", type=int, required=False, default=42)
    parser.add_argument("--number-iterations", action="store", type=int, required=False, default=1000)
    parser.add_argument("--result-database", action="store", type=pathlib.Path, required=False, default="results/sqlite.db")
    parser.add_argument("--load-attribution-file", action="store", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--save-attribution-file", action="store", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--random-threshold", action="store", type=float, required=False, default=0)
    parser.add_argument("--injection-type", action="store", type=str, choices=("activation", "weight"), required=True, default="activation")
    parser.add_argument("--sparse-target", action="store", type=str, choices=("index", "value"), required=False)
    # parser.add_argument("--devices", action="store", type=str, required=False, default="")

    return parser


def main(args=None):
    # TODO: improve the config handling
    parser = arg_parser()
    namespace = parser.parse_args(args)
    config = namespace.python_config.absolute()
    target_csv = namespace.target_csv.absolute()
    seed = namespace.seed
    iterations = namespace.number_iterations
    sqlite_db = namespace.result_database.absolute()

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    sys.path.append(str(config.parent))

    module_name = config.with_suffix("").name

    module = importlib.import_module(module_name)

    sys.path.pop()

    config_dict = module.config()

    trainer = config_dict["trainer"]
    datamodule = config_dict["datamodule"]
    model = config_dict["model"]

    injection_callback = get_generic_injection_callback(result_database=sqlite_db)
    trainer.callbacks.append(injection_callback)

    test_batch = next(datamodule.train_dataloader().__iter__())
    test_input = test_batch[flash.core.data.io.input.DataKeys.INPUT]
    test_target = test_batch[flash.core.data.io.input.DataKeys.TARGET]
    if namespace.sparse_target is not None:
        extra_injection_info = {"sparse_target": namespace.sparse_target}
    else:
        extra_injection_info = None
    sampling_model = enpheeph.helpers.importancesampling.ImportanceSampling(model=model, injection_type=namespace.injection_type, sensitivity_class="LayerConductance", test_input=test_input, random_threshold=namespace.random_threshold, seed=seed, extra_injection_info=extra_injection_info)
    if namespace.load_attribution_file is not None:
        sampling_model.load_attributions(namespace.load_attribution_file)
    else:
        sampling_model.generate_attributions(test_input=test_input, test_target=test_target)
        if namespace.save_attribution_file is not None:
            sampling_model.save_attributions(namespace.save_attribution_file)

    pytorch_mask_plugin = enpheeph.injections.plugins.mask.AutoPyTorchMaskPlugin()
    for i in range(iterations):
        sample = sampling_model.get_sample()
        print(f"iteration #{i}", sample)

        layer = sample["layer"]
        index = sample["index"]
        bit_index = sample["bit_index"]

        fault = enpheeph.injections.OutputPyTorchFault(
            location=enpheeph.utils.dataclasses.FaultLocation(
                module_name=layer,
                parameter_type=enpheeph.utils.enums.ParameterType.Activation,
                dimension_index={
                    enpheeph.utils.enums.DimensionType.Batch: ...,
                    enpheeph.utils.enums.DimensionType.Tensor: index,
                },
                bit_index=bit_index,
                bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
            ),
            low_level_torch_plugin=pytorch_mask_plugin,
            indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
                dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
            ),
        )
        monitor = enpheeph.injections.OutputPyTorchMonitor(
            location=enpheeph.utils.dataclasses.MonitorLocation(
                module_name=layer,
                parameter_type=enpheeph.utils.enums.ParameterType.Activation,
                dimension_index={
                    enpheeph.utils.enums.DimensionType.Tensor: ...,
                    enpheeph.utils.enums.DimensionType.Batch: ...,
                },
                bit_index=None,
            ),
            enabled_metrics=(
                enpheeph.utils.enums.MonitorMetric.ArithmeticMean
                | enpheeph.utils.enums.MonitorMetric.StandardDeviation
            ),
            storage_plugin=injection_callback.storage_plugin,
            move_to_first=False,
            indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
                dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
            ),
        )

        injection_callback.injection_handler.add_injections([fault, monitor])

        injection_callback.injection_handler.activate()
        # print(injection_callback.injection_handler.active_injections)
        result_injected = trainer.test(model, datamodule=datamodule)[0]

        if i == 0:
            injection_callback.injection_handler.activate()
            injection_callback.injection_handler.deactivate(
                [
                    inj
                    for inj in injection_callback.injection_handler.injections
                    if isinstance(inj, enpheeph.injections.abc.FaultABC)
                ]
            )
            # print(injection_callback.injection_handler.active_injections)
            result_baseline = trainer.test(model, datamodule=datamodule)[0]
        else:
            result_baseline = {k: float("NaN") for k in result_injected.keys()}

        if i == 0:
            injection_callback.injection_handler.activate()
            injection_callback.injection_handler.deactivate(
                [
                    inj
                    for inj in injection_callback.injection_handler.injections
                    if isinstance(inj, enpheeph.injections.abc.FaultABC)
                ]
            )
            # print(injection_callback.injection_handler.active_injections)
            trainer.test(model, datamodule=datamodule)

        # we need to remove all the injections to avoid stacking the faults
        injection_callback.injection_handler.remove_injections()

        with target_csv.open("a") as csv_file:
            if i == 0:
                csv_file.write("i,random_seed,random_threshold,timestamp,target_csv,")
                csv_file.write("injection_type,extra_injection_info,layer_name,index,bit_index,random,")
                csv_file.write(f"{','.join(k + '_baseline' for k in result_baseline.keys())},{','.join(k + '_injected' for k in result_injected.keys())}\n")
            csv_file.write(f"{i},")
            csv_file.write(f"{namespace.seed},")
            csv_file.write(f"{namespace.random_threshold},")
            csv_file.write(f"{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f%z')},")
            csv_file.write(f"{str(namespace.target_csv).replace(',', '---')},")
            csv_file.write(f"{namespace.injection_type},")
            csv_file.write(f"{tuple(sampling_model.extra_injection_info.items()) if sampling_model.extra_injection_info is not None else str(None)},")
            csv_file.write(f"{sample['layer']},")
            csv_file.write(f"{'-'.join(str(i) for i in sample['index'])},")
            csv_file.write(f"{sample['bit_index']},")
            csv_file.write(f"{sample['random']},")
            csv_file.write(f"{','.join(str(v) for v in result_baseline.values())},")
            csv_file.write(f"{','.join(str(v) for v in result_injected.values())}\n")

if __name__ == "__main__":
    main()
