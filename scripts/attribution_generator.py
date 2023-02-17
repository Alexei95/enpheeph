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

# we use conda enpheeph-scripts-dev

import pathlib
import shlex
import subprocess
import time

NUMBER_ITERATIONS = '1'
SEED = '1600'
CONFIG_FILES = [
    "configs/python/inference_pruned_image_classifier_resnet18_gtsrb.py",
    "configs/python/inference_image_classifier_vgg11_cifar10.py",
    "configs/python/inference_image_classifier_vgg11_gtsrb.py",
    # "configs/python/inference_pruned_semantic_segmenter_mobilenetv3_carla.py",
]
CHECKPOINT_FILES = {
    CONFIG_FILES[0]: [
        x.absolute()
        for x in pathlib.Path("results/sparse_results_ijcnn2023/trained_networks/resnet18_gtsrb/").glob("*.ckpt")
    ],
    CONFIG_FILES[1]: [
        x.absolute()
        for x in pathlib.Path("results/sparse_results_ijcnn2023/trained_networks/vgg11_cifar10/").glob("*.ckpt")
    ],
    CONFIG_FILES[2]: [
        x.absolute()
        for x in pathlib.Path("results/sparse_results_ijcnn2023/trained_networks/vgg11_gtsrb/").glob("*.ckpt")
    ],
    # CONFIG_FILES[1]: [
    #     x.absolute()
    #     for x in pathlib.Path("results/sparse_results_ijcnn2023/trained_networks/mobilenetv3_carla/").glob("*.ckpt")
    # ],
}
SAVE_ATTRIBUTION_PATH = "results/sparse_results_ijcnn2023/attributions/"
RESULT_DATABASE = "results/sparse_results_ijcnn2023/test/attributions_test.sqlite"
RANDOM_THRESHOLD = '1'
INJECTION_TYPE = "activation"
SPARSE_TARGETS = ["_"]
# INJECTION_TYPE = "weight"
# SPARSE_TARGETS = ["index", "value"]

def main():
    subprocess_command = [
        "python", "injector_script.py",
        "--seed", str(SEED),
        "--result-database", str(pathlib.Path(RESULT_DATABASE).absolute()),
        "--random-threshold", str(RANDOM_THRESHOLD),
        "--number-iterations", str(NUMBER_ITERATIONS),
        "--injection-type", str(INJECTION_TYPE),
    ]
    log_file = pathlib.Path(SAVE_ATTRIBUTION_PATH).absolute() / f"attribution_generator_log__{time.strftime('%Y_%m_%d__%H_%M_%S_%z')}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_hndl = log_file.open("a")
    grid_space = [(config, ckpt_file, sparse_target) for config in CONFIG_FILES for ckpt_file in CHECKPOINT_FILES[config] for sparse_target in SPARSE_TARGETS]
    for config, ckpt_file, sparse_target in grid_space:
        subprocess_command_1 = subprocess_command + ["--python-config", str(pathlib.Path(config).absolute())]
        subprocess_command_2 = subprocess_command_1 + ["--checkpoint-file", str(pathlib.Path(ckpt_file).absolute())]
        ckpt_file_path = pathlib.Path(ckpt_file)
        experiment_name = ckpt_file_path.parent.name
        ckpt_name = ckpt_file_path.with_suffix("").name.replace("=", "-")
        save_file = pathlib.Path(SAVE_ATTRIBUTION_PATH).absolute() / experiment_name / ckpt_name / ("_".join([sparse_target, INJECTION_TYPE, "seed", SEED]) + ".pt")
        save_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess_command_2 += ["--save-attribution-file", str(save_file)]
        if sparse_target != "_":
            subprocess_command_3 = subprocess_command_2 + ["--sparse-target", sparse_target]
        else:
            subprocess_command_3 = subprocess_command_2

        log_hndl.write(f"launching {subprocess_command_3}\n")
        log_hndl.flush()
        subprocess.run(
            shlex.split(" ".join(subprocess_command_3)),
            stdout=log_hndl, stderr=subprocess.STDOUT )
        log_hndl.flush()

if __name__ == "__main__":
    main()
