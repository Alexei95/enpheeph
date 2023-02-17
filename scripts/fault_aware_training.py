import pathlib
import sys
import time

import enpheeph
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import fault_aware_training_importance_sampling

FAULTS = {
    4: [
        enpheeph.utils.dataclasses.FaultLocation(
            module_name="linear_relu_stack.4",
            parameter_type=enpheeph.utils.enums.ParameterType.Weight,
            parameter_name="weight",
            dimension_index={
                enpheeph.utils.enums.DimensionType.Batch: ...,
                enpheeph.utils.enums.DimensionType.Tensor: (5, 404),
            },
            bit_index=(30, ),
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
        ),
        enpheeph.utils.dataclasses.FaultLocation(
            module_name="linear_relu_stack.2",
            parameter_type=enpheeph.utils.enums.ParameterType.Weight,
            parameter_name="weight",
            dimension_index={
                enpheeph.utils.enums.DimensionType.Batch: ...,
                enpheeph.utils.enums.DimensionType.Tensor: (95, 403),
            },
            bit_index=(30, ),
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
        ),
        enpheeph.utils.dataclasses.FaultLocation(
            module_name="linear_relu_stack.4",
            parameter_type=enpheeph.utils.enums.ParameterType.Weight,
            parameter_name="weight",
            dimension_index={
                enpheeph.utils.enums.DimensionType.Batch: ...,
                enpheeph.utils.enums.DimensionType.Tensor: (7, 380),
            },
            bit_index=(30, ),
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
        ),
        enpheeph.utils.dataclasses.FaultLocation(
            module_name="linear_relu_stack.2",
            parameter_type=enpheeph.utils.enums.ParameterType.Weight,
            parameter_name="weight",
            dimension_index={
                enpheeph.utils.enums.DimensionType.Batch: ...,
                enpheeph.utils.enums.DimensionType.Tensor: (90, 400),
            },
            bit_index=(30, ),
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.BitFlip,
        ),
    ],
}


def inject_fault(model, fault):
    target_layer = model
    for l in fault.module_name.split("."):
        target_layer = getattr(target_layer, l)

    def hook(module, input):
        with torch.no_grad():
            numpy_value = module.weight[fault.dimension_index[enpheeph.utils.enums.DimensionType.Tensor]].numpy()
            uint_dtype = numpy.dtype(f"u{str(numpy_value.dtype.itemsize)}")
            int_value = numpy_value.view(uint_dtype) ^ numpy.array(2 ** fault.bit_index[0], dtype=uint_dtype)
            new_value = int_value.view(numpy_value.dtype)
            # print(numpy_value, new_value)
            module.weight[fault.dimension_index[enpheeph.utils.enums.DimensionType.Tensor]] = torch.from_numpy(numpy.array(new_value))

    handle = target_layer.register_forward_pre_hook(hook)

    return handle

if __name__ == "__main__":
    fault_aware_training_importance_sampling.set_seed(seed=1600)
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

    model = fault_aware_training_importance_sampling.NeuralNetwork()
    model.init_weights()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # csv_path = pathlib.Path("results/sparse_results_ijcnn2023/fault_aware_training/injection_training/") / (time.strftime(fault_aware_training_importance_sampling.TIME_FORMAT) + ".csv")
    log_path = pathlib.Path("results/sparse_results_ijcnn2023/fault_aware_training/injection_training/") / (time.strftime(fault_aware_training_importance_sampling.TIME_FORMAT) + "_log.txt")
    # csv_file = csv_path.open("w")
    log_file = log_path.open("w")


    duplicate_stream = fault_aware_training_importance_sampling.ConsoleFileOutput(file=log_file, stream=sys.stdout)

    sys.stdout = duplicate_stream

    handles = []
    for t in range(15):
        print(f"Epoch {t}\n-------------------------------")
        fault_aware_training_importance_sampling.train_loop(train_dataloader, model, loss_fn, optimizer)
        # for h in handles:
        #     h.remove()
        fault_aware_training_importance_sampling.test_loop(test_dataloader, model, loss_fn)
        for e, faults in FAULTS.items():
            if t == e:
                print("baseline test accuracy")
                fault_aware_training_importance_sampling.test_loop(test_dataloader, model, loss_fn)
                for fault in faults:
                    print(f"Epoch {e}, injecting fault {fault}")
                    handles.append(inject_fault(model, fault))
                    print("faulty test accuracy")
                    fault_aware_training_importance_sampling.test_loop(test_dataloader, model, loss_fn)

    print("faulty test accuracy")
    fault_aware_training_importance_sampling.test_loop(test_dataloader, model, loss_fn)
    print("removing faults")
    for h in handles:
        h.remove()
    print("no fault accuracy")
    fault_aware_training_importance_sampling.test_loop(test_dataloader, model, loss_fn)

    # torch.save(model, "model.pt")
    print("Done!")
