import asyncio
import argparse
import logging
import pathlib
import shlex
import sys

# INJECTOR_SCRIPT_PATH = pathlib.Path("injector_script.py").absolute()
# CSV_FILE_NAME_TEMPLATE = "device_{}_seed_{}_table.csv"
# OUTPUT_STDERR_FILE_NAME_TEMPLATE = "device_{}_seed_{}_output_stderr.txt"
# OUTPUT_STDOUT_FILE_NAME_TEMPLATE = "device_{}_seed_{}_output_stdout.txt"
# DATABASE_FILE_NAME_TEMPLATE = "device_{}_seed_{}_database.sqlite"
# TASK_MANAGER_OUTPUT_FILE_NAME_TEMPLATE = "task_manager_output.txt"
# RESULT_DIRECTORY_TEMPLATE = "results_devices_{}_starting_seed_{}"
INJECTOR_SCRIPT_PATH = pathlib.Path("injector_script.py").absolute()
CSV_FILE_NAME_TEMPLATE = "table.csv"
OUTPUT_STDERR_FILE_NAME_TEMPLATE = "output_stderr.txt"
OUTPUT_STDOUT_FILE_NAME_TEMPLATE = "output_stdout.txt"
DATABASE_FILE_NAME_TEMPLATE = "database.sqlite"
TASK_MANAGER_OUTPUT_FILE_NAME_TEMPLATE = "task_manager_output.txt"
RESULT_DIRECTORY_TEMPLATE = "results_injection_type_{}_devices_{}_starting_seed_{}_random_threshold_{}"
ITERATION_RESULT_TEMPLATE = "device_{}_seed_{}"

LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--python-config", action="store", type=pathlib.Path, required=True)
    parser.add_argument("--starting-seed", action="store", type=int, required=False, default=42)
    parser.add_argument("--number-iterations", action="store", type=int, required=False, default=1000)

    parser.add_argument("--result-folder", action="store", type=pathlib.Path, required=False, default="results")
    parser.add_argument("--devices", action="store", type=str, required=False, default="")
    parser.add_argument("--load-attribution-file", action="store", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--random-threshold", action="store", type=float, required=False, default=0)
    parser.add_argument("--injection-type", action="store", type=str, choices=("activation", "weight"), required=True, default="activation")
    parser.add_argument("--sparse-target", action="store", type=str, choices=("index", "value"), required=False)

    return parser


async def run_injector_script(device, seed, namespace):
    result_directory = namespace.result_folder.absolute() / RESULT_DIRECTORY_TEMPLATE.format(namespace.injection_type, namespace.devices, namespace.starting_seed, namespace.random_threshold) / ITERATION_RESULT_TEMPLATE.format(device, seed)
    result_directory.mkdir(parents=True, exist_ok=True)

    iterations = namespace.number_iterations
    target_csv = result_directory / CSV_FILE_NAME_TEMPLATE
    database = result_directory / DATABASE_FILE_NAME_TEMPLATE
    output_stdout = result_directory / OUTPUT_STDOUT_FILE_NAME_TEMPLATE
    output_stderr = result_directory / OUTPUT_STDERR_FILE_NAME_TEMPLATE
    command_env = f"CUDA_VISIBLE_DEVICES={device}"
    command_exec = str(sys.executable)
    command_args = f"{str(INJECTOR_SCRIPT_PATH)} --python-config {str(namespace.python_config.absolute())} --seed {seed} --target-csv {str(target_csv)} --number-iterations {iterations} --result-database {str(database)} --random-threshold {namespace.random_threshold} --injection-type {namespace.injection_type}"
    if namespace.load_attribution_file is not None:
        command_args += f" --load-attribution-file {namespace.load_attribution_file}"
    if namespace.sparse_target is not None:
        command_args += f" --sparse-target {namespace.sparse_target}"

    logging.getLogger(__name__).info(f"device={device}, seed={seed}, running '{command_env} {command_exec} {command_args}'")

    with open(output_stdout, "w") as stdout, open(output_stderr, "w") as stderr:
        process = await asyncio.create_subprocess_exec(command_exec, *shlex.split(command_args), stdout=stdout, stderr=stderr, env={"CUDA_VISIBLE_DEVICES": device})
        # process = await asyncio.create_subprocess_exec(command_exec, *shlex.split(command_args), env={"CUDA_VISIBLE_DEVICES": str(device)})

    await process.wait()

    logging.getLogger(__name__).info(f"device={device}, seed={seed}, return code={process.returncode}")


async def main(args=None):
    parser = get_arg_parser()
    namespace = parser.parse_args(args)

    result_directory = namespace.result_folder / RESULT_DIRECTORY_TEMPLATE.format(namespace.injection_type, namespace.devices, namespace.starting_seed, namespace.random_threshold)
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

    if namespace.devices == "":
        devices = [""]
    else:
        devices = namespace.devices.split(",")
    seed = namespace.starting_seed
    await asyncio.gather(*[run_injector_script(d, seed + s, namespace) for s, d in enumerate(devices)])


if __name__ == "__main__":
    asyncio.run(main())
