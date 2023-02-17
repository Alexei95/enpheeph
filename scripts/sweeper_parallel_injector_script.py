import argparse
import asyncio
import itertools
import logging
import pathlib
import shlex
import sys
import time
import typing

TIME_FORMAT = '%Y_%m_%d__%H_%M_%S_%z'
# INJECTOR_SCRIPT_PATH = pathlib.Path("injector_script.py").absolute()
# CSV_FILE_NAME_TEMPLATE = "device_{}_seed_{}_table.csv"
# OUTPUT_STDERR_FILE_NAME_TEMPLATE = "device_{}_seed_{}_output_stderr.txt"
# OUTPUT_STDOUT_FILE_NAME_TEMPLATE = "device_{}_seed_{}_output_stdout.txt"
# DATABASE_FILE_NAME_TEMPLATE = "device_{}_seed_{}_database.sqlite"
# TASK_MANAGER_OUTPUT_FILE_NAME_TEMPLATE = "task_manager_output.txt"
# RESULT_DIRECTORY_TEMPLATE = "results_devices_{}_starting_seed_{}"
CWD = pathlib.Path(__file__).absolute().parent
PARALLEL_INJECTOR_SCRIPT_PATH = CWD / "parallel_injector_script.py"
TASK_MANAGER_OUTPUT_FILE_NAME_TEMPLATE = "sweeper_output.txt"
RESULT_DIRECTORY_TEMPLATE = "results_iterations_{}_timeout_{}_device_{}__" + time.strftime(TIME_FORMAT)
LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


BASE_EXPERIMENTS = [
    # resnet18 gtsrb
    {
        "python_config": CWD / "configs/python/inference_pruned_image_classifier_resnet18_gtsrb.py",
        "checkpoint_file": CWD / "results/sparse_results_ijcnn2023/trained_networks/resnet18_gtsrb/epoch=29-step=19980_0_pruning.ckpt",
        "base_directory_load_attribution_file": CWD / "results/sparse_results_ijcnn2023/attributions/resnet18_gtsrb/epoch-29-step-19980_0_pruning/",
    },
    # resnet18 cifar10
    {
        "python_config": CWD / "configs/python/inference_pruned_image_classifier_resnet18_cifar10.py",
        "checkpoint_file": CWD / "results/sparse_results_ijcnn2023/trained_networks/resnet18_cifar10/epoch=30-step=38750_0_pruning.ckpt",
        "base_directory_load_attribution_file": CWD / "results/sparse_results_ijcnn2023/attributions/resnet18_cifar10/epoch-30-step-38750_0_pruning/",
    },
    # vgg11 gtsrb
    {
        "python_config": CWD / "configs/python/inference_image_classifier_vgg11_gtsrb.py",
        "checkpoint_file": CWD / "results/sparse_results_ijcnn2023/trained_networks/vgg11_gtsrb/epoch=29-step=19980.ckpt",
        "base_directory_load_attribution_file": CWD / "results/sparse_results_ijcnn2023/attributions/vgg11_gtsrb/epoch-29-step-19980/",
    },
    # vgg11 cifar10
    {
        "python_config": CWD / "configs/python/inference_image_classifier_vgg11_cifar10.py",
        "checkpoint_file": CWD / "results/sparse_results_ijcnn2023/trained_networks/vgg11_cifar10/epoch=4-step=6250.ckpt",
        "base_directory_load_attribution_file": CWD / "results/sparse_results_ijcnn2023/attributions/vgg11_cifar10/epoch-4-step-6250/",
    },
]
BASE_CONFIGURATIONS = {
    "injection_type": ["weight", "activation"],
    "bit_weighting": ["gradient", "exponential", "linear", "random"],
    "random_threshold": [0, 1],
    "approximate_activation_gradient_value": [True],
    # "starting_seed": [1610, 1611,],
    # "starting_seed": [1612, 1613,],
    "starting_seed": [1614, 1615,],
}
EXPERIMENTS = []
for base_config, inj_type, bit_w, rand_thr, approx_grad_act, seed in itertools.product(BASE_EXPERIMENTS, *BASE_CONFIGURATIONS.values()):
    config = base_config.copy()
    if inj_type == "activation":
        attr_file = config["base_directory_load_attribution_file"] / "__activation_seed_1600.pt"
    elif inj_type == "weight":
        attr_file = config["base_directory_load_attribution_file"] / "index_weight_seed_1600.pt"
    else:
        raise ValueError()
    config.update({
        "injection_type": inj_type,
        "bit_weighting": bit_w,
        "random_threshold": rand_thr,
        "approximate_activation_gradient_value": approx_grad_act,
        "load_attribution_file": attr_file,
        "starting_seed": seed,
    })
    EXPERIMENTS.append(config)



# taken from more-itertools recipe in Python itertools docs
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch might be shorter."
    # batch('ABCDEFG', 3) -> ABC DEF G
    if n < 1:
        raise ValueError("'n' must be at least one")
    it = iter(iterable)
    while (batch := tuple(itertools.islice(it, n))):
        yield batch
def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')

def process_command_line(namespace: typing.Dict[str, str]):
    command_line = (
        f" {shlex.quote(str(PARALLEL_INJECTOR_SCRIPT_PATH))} "
        f"--python-config {shlex.quote(str(namespace['python_config']))} "
        f"--checkpoint-file {shlex.quote(str(namespace['checkpoint_file']))} "
        f"--starting-seed {shlex.quote(str(namespace['starting_seed']))} "
        f"--result-folder {shlex.quote(str(namespace['result_folder']))} "
        f"--devices {shlex.quote(str(namespace['devices']))} "
        f"--result-folder {shlex.quote(str(namespace['result_folder']))} "
        f"--injection-type {shlex.quote(str(namespace['injection_type']))} "
        f"--random-threshold {shlex.quote(str(namespace['random_threshold']))} "
    )
    if (value := namespace.get("load_attribution_file", None)) is not None:
        command_line += f"--load-attribution-file {shlex.quote(str(value))} "
    if (value := namespace.get("number_iterations", None)) is not None:
        command_line += f"--number-iterations {shlex.quote(str(value))} "
    if (value := namespace.get("sparse_target", None)) is not None:
        command_line += f"--sparse-target {shlex.quote(str(value))} "
    if (value := namespace.get("checkpoint_file", None)) is not None:
        command_line += f"--checkpoint-file {shlex.quote(str(value))} "
    if (value := namespace.get("timeout", None)) is not None:
        command_line += f"--timeout {shlex.quote(str(value))} "
    if (value := namespace.get("bit_weighting", None)) is not None:
        command_line += f"--bit-weighting {shlex.quote(str(value))} "
    if (value := namespace["approximate_activation_gradient_value"]) is not None:
        if value is True:
            command_line += f" --approximate-activation-gradient-value "
        elif value is False:
            command_line += f" --no-approximate-activation-gradient-value "
        else:
            raise ValueError("how can it possibly be not True and not False?")

    return command_line

def get_arg_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--seed", action="store", type=int, required=False, default=42)
    parser.add_argument("--number-iterations", action="store", type=int, required=False, default=1000)
    parser.add_argument("--timeout", action="store", type=int, default=None, required=False)
    parser.add_argument("--result-folder", action="store", type=pathlib.Path, required=False, default="results")
    parser.add_argument("--devices", action="store", type=str, required=False, default="")
    return parser


async def run_parallel_injector_script(command_lines: typing.List[str], command_exec="python"):
    for command_line in command_lines:
        logging.getLogger(__name__).info(f"running command_exec={command_exec}, command_line={command_line}")

        process = await asyncio.create_subprocess_exec(command_exec, *shlex.split(command_line))

        await process.wait()

        logging.getLogger(__name__).info(f"finished command_exec={command_exec}, command_line={command_line}, return code={process.returncode}")


async def main(experiments: typing.List[typing.Dict[str, str]], args=None):
    parser = get_arg_parser()
    namespace = parser.parse_args(args)
    result_directory_name = RESULT_DIRECTORY_TEMPLATE.format(namespace.number_iterations, namespace.timeout, namespace.devices)
    result_directory = namespace.result_folder.absolute() / result_directory_name
    result_directory.mkdir(parents=True, exist_ok=True)

    output = result_directory / TASK_MANAGER_OUTPUT_FILE_NAME_TEMPLATE

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=LOGGER_FORMAT, datefmt=LOGGER_DATE_FORMAT)
    file_handler = logging.FileHandler(output)
    file_handler.setFormatter(fmt=formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    logger.info(f"parsed arguments: {str(namespace)}")
    logger.info(f"experiments to run: {str(experiments)}")

    if namespace.devices == "":
        devices = [""]
    else:
        devices = namespace.devices.split(",")
    command_lines_by_device = {d: [] for d in devices}
    for experiment_group in grouper(experiments, len(devices), incomplete="fill", fillvalue={}):
        for device, experiment in zip(devices, experiment_group):
            if experiment == {}:
                continue
            complete_experiment = experiment.copy()
            complete_experiment.update(namespace.__dict__)
            complete_experiment.update({"devices": device, "result_folder": result_directory})
            command_line = process_command_line(complete_experiment)
            command_lines_by_device[device].append(command_line)

    tasks = {d: asyncio.create_task(run_parallel_injector_script(c)) for d, c in command_lines_by_device.items()}

    await asyncio.gather(*[c for c in tasks.values()])


if __name__ == "__main__":
    # print(len(EXPERIMENTS))
    asyncio.run(main(experiments=EXPERIMENTS))
