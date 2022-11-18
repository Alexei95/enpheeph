import itertools
import os
import pathlib
import random

import captum
import captum.attr
import torch
import torchinfo


# due to the sensitivity being based on output, this importance sampling works only
# on output neurons for now
class ImportanceSampling(object):
    RANDOM_THRESHOLD = 0.1
    SEED = 42
    SENSITIVITY_CLASSES = {
        # layer conductance works on activations, e.g. output layer neurons
        "LayerConductance": captum.attr.LayerConductance,
    }
    INJECTION_TYPES = {
        "activation",
        "weight",
    }
    EXTRA_INJECTION_INFO_PAIRS = [
        ("sparse_target", "index"),
        ("sparse_target", "value"),
    ]
    # we assume the test inputs/tagets and model are on the same device
    def __init__(self, model: torch.nn.Module, test_input, sensitivity_class: str, injection_type: str, random_threshold=RANDOM_THRESHOLD, seed=SEED, extra_injection_info=None):
        self.sensitivity_class_name = sensitivity_class
        try:
            self.sensitivity_class = self.SENSITIVITY_CLASSES[sensitivity_class]
        except KeyError:
            raise ValueError(f"only allowed sensitivity class strings are {','.join(self.SENSITIVITY_CLASSES.keys())}")
        if injection_type not in self.INJECTION_TYPES:
            raise ValueError(f"only allowed injection type strings are {','.join(self.INJECTION_TYPES)}")
        else:
            self.injection_type = injection_type

        if extra_injection_info is not None:
            for k, v in extra_injection_info.items():
                if (k, v) not in self.EXTRA_INJECTION_INFO_PAIRS:
                    raise ValueError(f"only allowed pairs in 'extra_injection_info' are {self.EXTRA_INJECTION_INFO_PAIRS}")
        self.extra_injection_info = extra_injection_info

        # we remove the batch dimension
        self.summary = torchinfo.summary(model=model, input_size=test_input.size()[1:], batch_dim=0, verbose=0, device="cpu")

        self.random_generator = random.Random(x=seed)
        self.random_threshold = random_threshold

    @classmethod
    def _init_only_attributions():
        pass

    def load_attributions(self, save_file: os.PathLike):
        save_file = pathlib.Path(save_file)
        data = torch.load(str(save_file))
        self.summary = data["summary"]
        self.injection_type = data["injection_type"]
        self.sensitivity_class_name = data["sensitivity_class"]
        self.sensitivity = self.SENSITIVITY_CLASSES[self.sensitivity_class_name]
        self.leaf_layers = data["leaf_layers"]
        self.sensitivity_layers = data["sensitivity_layers"]
        self.extra_injection_info = data["extra_injection_info"]

    def save_attributions(self, save_file: os.PathLike):
        save_file = pathlib.Path(save_file)
        data = {"injection_type": self.injection_type, "sensitivity_class": self.sensitivity_class_name, "leaf_layers": self.leaf_layers, "sensitivity_layers": self.sensitivity_layers, "extra_injection_info": self.extra_injection_info, "summary": self.summary}
        torch.save(data, str(save_file))

    def generate_attributions(self, test_input, test_target=None):
        leaf_layers = [l for l in self.summary.summary_list if l.is_leaf_layer]

        self.leaf_layers = []
        self.sensitivity_layers = {}
        # sensitivity_by_index_layers = {}
        for layer in leaf_layers:
            if self.injection_type == "activation":
                sensitivity_instance = self.sensitivity_class(forward_func=self.summary.summary_list[0].module, layer=layer.module)
                # we need try as when using lightning-flash some models have the metrics
                # counted as leaf_layers, and they might create problems, so they must be
                # filtered out
                try:
                    # we average the attributions over the batch dimension
                    # no keepdim needed
                    sensitivity = sensitivity_instance.attribute(inputs=test_input, target=test_target).mean(dim=0)
                except AssertionError:
                    continue

                # it cannot be computed here as it would take too much RAM
                # indices_product = itertools.product(*[range(dim) for dim in sensitivity.size()])
                # sensitivity_by_index = []
                # for index in indices_product:
                #     sensitivity_by_index.append({"index": index, "value": sensitivity[index].item})
                # sensitivity_by_index_layers[layer] = sensitivity_by_index
            elif self.injection_type == "weight":
                with torch.set_grad_enabled(True):
                    temp_output = self.summary.summary_list[0].module(test_input)
                temp_output = temp_output.flatten()
                try:
                    sensitivity = torch.zeros_like(layer.module.weight)
                except AttributeError:
                    continue
                for i in range(temp_output.nelement()):
                    layer.module.zero_grad()
                    temp_output[i].backward(retain_graph=True)
                    sensitivity += layer.module.weight.grad
                    layer.module.zero_grad()
                sensitivity = sensitivity.cpu()
            else:
                raise ValueError("invalid injection type")

            self.leaf_layers.append(layer)
            self.sensitivity_layers[layer] = sensitivity

        # self.sensitivity_by_index_layers = sensitivity_by_index_layers

    def get_sample(self):
        # we sample a layer among the ones in the model
        layer_sample = self.random_generator.sample(population=self.leaf_layers, k=1)[0]
        layer_name_list = []
        l = layer_sample
        while l is not None:
            layer_name_list.append(l.var_name)
            l = l.parent_info
        # we reverse the list, so we can join it from parent to child and not
        # vice versa
        # additionally with -2 we remove the last element before reversal, which is
        # the name of the network itself and it is not needed
        layer_name_list = layer_name_list[-2::-1]
        layer_name = ".".join(layer_name_list)

        # we choose whether the injection will be random or based on sensitivity
        r = self.random_generator.random()
        if r < self.random_threshold:
            randomness = True
            # we need to provide the weights for the bit index
            # we assume the target is FP32, where 31 is the sign bit
            # then 30-23 are exponent bits in MSB->LSB order and finally the mantissa
            # in the same order
            # for random sampling we use a completely random choice
            bit_index = self.random_generator.choices(
                list(range(0, 32)),
            )[0]

            if self.injection_type == "activation":
                # we remove the first size as it represents the batch size
                target_size = layer_sample.output_size[1:]
            elif self.injection_type == "weight":
                target_size = layer_sample.module.weight.size()
            else:
                raise ValueError("unsupported injection")

            index = []
            for dim in target_size:
                index.append(self.random_generator.randint(0, dim - 1))
            index_sample = list(index)
        else:
            randomness = False
            # we need to provide the weights for the bit index
            # we assume the target is FP32, where 31 is the sign bit
            # then 30-23 are exponent bits in MSB->LSB order and finally the mantissa
            # in the same order
            # we consider each bit to have a weight corresponding to its position, as this
            # somewhat represents the importance of each bit
            # the weights are shifted by one so that we have a non-zero weight for 0 as well
            bit_index = self.random_generator.choices(
                list(range(0, 32)),
                # linear weighting
                # weights=list(range(1, 33))
                # bit-correct weighting
                weights=[2 ** x for x in range(0, 32)]
            )[0]

            sensitivity = self.sensitivity_layers[layer_sample]
            # print(sensitivity.size())
            indices_product = itertools.product(*[range(dim) for dim in sensitivity.size()])
            sensitivity_by_index = []
            for index in indices_product:
                sensitivity_by_index.append({"index": index, "value": sensitivity[index].item()})
            indices = [x["index"] for x in sensitivity_by_index]
            # we use abs as some attributions might be negative, and we are interested
            # in their magnitudes
            weights = [abs(x["value"]) for x in sensitivity_by_index]
            index_sample = list(self.random_generator.choices(population=indices, weights=weights, k=1)[0])

        if self.extra_injection_info is None:
            sparse_target = None
        else:
            sparse_target = self.extra_injection_info["sparse_target"]

        if sparse_target is not None:
            if self.injection_type != "weight":
                raise ValueError("unsupported sparse injection, only weight is supported")
            # we convert the weight from the chosen layer to sparse as the indices
            # are in a list so we need to get the index of the indices in the list
            # we get the tensor of the indices
            index_tensor = layer_sample.module.weight.to_sparse().indices()
            # as the indices are in rows of the coordinates, each row is a list of
            # all the coordinates, each column is a coordinate for a single value
            # so we need to flip it to match them
            index_sample = index_tensor.T.tolist().index(index_sample)
            # for the index we need to also select the proper coordinate
            if sparse_target == "index":
                # the index sample in this case is the second index, as the first should
                # be the row of the index to be hit by the fault
                subindex_sample = self.random_generator.randint(0, index_tensor.size()[0])
                index_sample = [subindex_sample, index_sample]
            elif sparse_target == "value":
                index_sample = [index_sample]

        return {"layer": layer_name, "index": index_sample, "bit_index": bit_index, "random": randomness, "injection_type": self.injection_type, "sparse_target": sparse_target}
