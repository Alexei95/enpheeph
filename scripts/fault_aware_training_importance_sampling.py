import copy
import itertools
import pathlib
import random
import sys
import time

import enpheeph
import enpheeph.helpers.importancesampling
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

TIME_FORMAT = '%Y_%m_%d__%H_%M_%S_%z'

def set_seed(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ConsoleFileOutput:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.file.flush()
        self.stream.flush()


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    def init_weights(self):
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    current = 0
    correct = 0
    total_loss = 0

    model = model.train(mode=True)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        correct_batch = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct += correct_batch
        current += X.size()[0]

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            loss = loss.item()
            print(f"Accuracy (current batch): {correct_batch / len(X):>7f}, loss (current batch): {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return {"train_accuracy": correct / size, "train_loss": total_loss / len(dataloader)}


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model = model.train(mode=False)

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    return {"test_accuracy": correct, "test_loss": test_loss}


def init_importance_sampling(model, dataset, random_neuron=True, seed=42, bit_weighting="random", importancesampling=None, injection_type="weight", extra_injection_info={}):
    starting_time = time.time_ns()
    test_input, test_target = next(iter(dataset))
    # print(test_input.size(), test_target.size())
    extra_injection_info.update({"bit_weighting": bit_weighting})
    sampling_model = enpheeph.helpers.importancesampling.ImportanceSampling(model=model, injection_type=injection_type, sensitivity_class="LayerConductance", test_input=test_input, random_threshold=float(random_neuron), seed=seed, extra_injection_info=extra_injection_info)
    if importancesampling is None or sampling_model.injection_type != importancesampling.injection_type:
        sampling_model.generate_attributions(test_input=test_input, test_target=test_target)
    else:
        sampling_model.copy_attributions(importancesampling)
    print(f"attribution generation: {time.time_ns() - starting_time}ns")
    return sampling_model


def fault_injection(model, dataset, loss_fn, accuracy_drop_threshold, random_neuron=True, bit_weighting=None, seed=None, iteration_limit=None, epoch=None, baseline=None, importancesampling=None, accuracy_drop_lower_bound=None, n_faults=1, weight_or_activation_flag=None):
    if seed is None:
        seed = 1600
    # must be run in testing mode, so without grad
    # we cannot modify the running values with enpheeph yet, as it requires going
    # through numpy/cupy
    # we can do the analysis with enpheeph, and hardcode the fault later on
    # we can use a forward hook

    model = model.train(mode=False)

    # inputs: model, dataset, accuracy drop threshold

    # we compute baseline
    if baseline is None:
        baseline = test_loop(dataloader=dataset, model=model, loss_fn=loss_fn)

    base_payload_element = {"random_neuron": random_neuron, "bit_weighting": bit_weighting, "iteration": 0, "execution_time_ns": 0, "accuracy_drop_threshold": accuracy_drop_threshold, "fault_location": None, "test_info": {}, "test_info_baseline": baseline, "training_epoch": epoch, "seed": seed, "iteration_limit": iteration_limit, "iteration_limit_reached": False, "accuracy_drop_lower_bound": accuracy_drop_lower_bound, "accuracy_drop_lower_bound_reached": False, "n_faults": n_faults, "current_fault_index": None, "weight_or_activation_flag": weight_or_activation_flag}
    payload = []
    if accuracy_drop_lower_bound is not None and baseline["test_accuracy"] - acc_drop <= accuracy_drop_lower_bound:
        payload = [copy.deepcopy(base_payload_element)]
        payload[-1]["accuracy_drop_lower_bound_reached"] = True
        csv_payload = "\n".join(["|".join([str(x) for x in p.values()]) for p in payload])
        print(payload)
        print(csv_payload)
        return payload, csv_payload


    # enpheeph fault injection
    # storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
    #     db_url="sqlite:///" + str(pathlib.Path(".").absolute() / "result.sqlite")
    # )
    pytorch_mask_plugin = enpheeph.injections.plugins.mask.AutoPyTorchMaskPlugin()
    pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()

    if weight_or_activation_flag is None or weight_or_activation_flag == "weight":
        injection_type = "weight"
        extra_injection_info = {}
    elif weight_or_activation_flag == "activation":
        injection_type = "activation"
        extra_injection_info = {"approximate_activation_gradient_value": True}
    else:
        raise ValueError("'weight_or_activation_flag' can only be None, 'weight', or 'activation'")

    sampling_model = init_importance_sampling(model=model, dataset=dataset, random_neuron=random_neuron, seed=seed, bit_weighting=bit_weighting, importancesampling=importancesampling, injection_type=injection_type, extra_injection_info=extra_injection_info)

    starting_time = time.time_ns()

    iteration = 0
    produced_faults = 0
    while 1:
        sample = sampling_model.get_sample()
        print(f"iteration {iteration}", sample)

        layer = sample["layer"]
        index = sample["index"]
        bit_index = sample["bit_index"]

        injection_handler = enpheeph.handlers.InjectionHandler(
            injections=[],
            library_handler_plugin=pytorch_handler_plugin,
        )
        if injection_type == "weight":
            fault_class = enpheeph.injections.WeightPyTorchFault
            parameter_type = enpheeph.utils.enums.ParameterType.Weight
            parameter_name = "weight"
        elif injection_type == "activation":
            fault_class = enpheeph.injections.OutputPyTorchFault
            parameter_type = enpheeph.utils.enums.ParameterType.Activation
            parameter_name = None
        else:
            raise ValueError("impossible")

        fault = fault_class(
            location=enpheeph.utils.dataclasses.FaultLocation(
                module_name=layer,
                parameter_type=parameter_type,
                parameter_name=parameter_name,
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
        # the storage requires the proper session/experiment setup
        # TODO: create a EmptyStorage to be used here without saving anything
        # TODO: create a PrintStorage to directly print the stuff
        # monitor = enpheeph.injections.OutputPyTorchMonitor(
        #     location=enpheeph.utils.dataclasses.MonitorLocation(
        #         module_name=layer,
        #         parameter_type=enpheeph.utils.enums.ParameterType.Activation,
        #         dimension_index={
        #             enpheeph.utils.enums.DimensionType.Tensor: ...,
        #             enpheeph.utils.enums.DimensionType.Batch: ...,
        #         },
        #         bit_index=None,
        #     ),
        #     enabled_metrics=(
        #         enpheeph.utils.enums.MonitorMetric.ArithmeticMean
        #         | enpheeph.utils.enums.MonitorMetric.StandardDeviation
        #     ),
        #     storage_plugin=storage_plugin,
        #     move_to_first=False,
        #     indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
        #         dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        #     ),
        # )

        # injection_handler.add_injections([fault, monitor])
        injection_handler.add_injections([fault])

        with torch.no_grad():
            injection_handler.activate()
            model = injection_handler.setup(model)
            # print(injection_callback.injection_handler.active_injections)
            results = test_loop(dataset, model, loss_fn)
            model = injection_handler.teardown(model)
            injection_handler.remove_injections()

        reached_it_limit_flag = iteration_limit is not None and iteration >= iteration_limit
        reached_acc_flag = baseline["test_accuracy"] - results["test_accuracy"] >= accuracy_drop_threshold
        payload_element = copy.deepcopy(base_payload_element)
        payload_element.update({"iteration": iteration, "execution_time_ns": time.time_ns() - starting_time, "fault_location": sample, "test_info": results, "iteration_limit_reached": reached_it_limit_flag, "injection_type": injection_type})
        if reached_acc_flag or reached_it_limit_flag:
            produced_faults += 1
            payload_element.update({"current_fault_index": produced_faults})
            payload.append(payload_element)
            iteration = -1
            if produced_faults >= n_faults:
                print(payload)
                csv_payload = "\n".join(["|".join([str(x) for x in p.values()]) for p in payload])
                print(csv_payload)
                return payload, csv_payload

        iteration += 1


# 69 epochs with 0.01 learning rate are good to avoid overfitting, seed 1600
# 60 epochs with 0.01 learning rate are good to avoid overfitting, it starts overfitting
EPOCHS = 100
LEARNING_RATE = 0.01
ITERATION_LIMIT = 50  # per fault, not total per experiment, easier to show
N_FAULTS = 3
seed = 1600
SEEDS = [1601, 1602, 1603]
accuracy_drop_thresholds = list(map(lambda x: x / 100, range(-5, 105, 5)))[::-1]
ACCURACY_DROP_LOWER_BOUND = 0.1
WEIGHT_OR_ACTIVATION_FLAGS = ["weight", "activation"]
# WEIGHT_OR_ACTIVATION_FLAGS = ["activation"]
# WEIGHT_OR_ACTIVATION_FLAGS = ["weight"]
# print(accuracy_drop_thresholds)
random_neuron_configs = [False, True]
bit_weighting_configs = ["random", "linear", "exponential", "gradient"]

if __name__ == "__main__":
    set_seed(seed=seed)
    training_data = datasets.FashionMNIST(
        root="/shared/ml/datasets/vision/FashionMNIST/",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="/shared/ml/datasets/vision/FashionMNIST/",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    model = NeuralNetwork()
    model.init_weights()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    csv_path = pathlib.Path("results/sparse_results_ijcnn2023/fault_aware_training") / (time.strftime(TIME_FORMAT) + ".csv")
    log_path = pathlib.Path("results/sparse_results_ijcnn2023/fault_aware_training") / (time.strftime(TIME_FORMAT) + "_log.txt")
    csv_file = csv_path.open("w")
    log_file = log_path.open("w")


    duplicate_stream = ConsoleFileOutput(file=log_file, stream=sys.stdout)

    sys.stdout = duplicate_stream

    payload, csv_payload = fault_injection(model=model, dataset=test_dataloader, loss_fn=loss_fn, accuracy_drop_threshold=0.00, random_neuron=False, bit_weighting="exponential", seed=0, iteration_limit=1)
    csv_file.write("|".join([str(k) for k in payload[0].keys()]) + "\n")
    csv_file.flush()

    for t in range(EPOCHS):
        print(f"Epoch {t}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        baseline = test_loop(test_dataloader, model, loss_fn)
        sampling_models = {}
        for flag in WEIGHT_OR_ACTIVATION_FLAGS:
            sampling_models[flag] = init_importance_sampling(model, test_dataloader, injection_type=flag)
        for acc_drop, random_neuron, bit_weighting, seed, weight_or_activation_flag in itertools.product(accuracy_drop_thresholds, random_neuron_configs, bit_weighting_configs, SEEDS, WEIGHT_OR_ACTIVATION_FLAGS):
            print(acc_drop, random_neuron, bit_weighting, seed, weight_or_activation_flag)
            payload, csv_payload = fault_injection(model=model, dataset=test_dataloader, loss_fn=loss_fn, accuracy_drop_threshold=acc_drop, random_neuron=random_neuron, bit_weighting=bit_weighting, seed=seed, iteration_limit=ITERATION_LIMIT, epoch=t, baseline=baseline, importancesampling=sampling_models[weight_or_activation_flag], accuracy_drop_lower_bound=ACCURACY_DROP_LOWER_BOUND, n_faults=N_FAULTS, weight_or_activation_flag=weight_or_activation_flag)
            csv_file.write(csv_payload + "\n")
            csv_file.flush()

    # torch.save(model, "model.pt")
    print("Done!")
