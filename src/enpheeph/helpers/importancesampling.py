import copy
import itertools
import os
import pathlib
import random
import sys

import captum
import captum.attr
import numpy
import torch
import torchinfo

EXPERIMENT_CODE_TEMPLATE = "{}B{}N{}"
BIT_WEIGHTING_CODE_CONVERSION = {
    "gradient": "G",
    "exponential": "E",
    "linear": "L",
    "random": "R",
}
NEURON_IMPORTANCE_CONVERSION = {
    False: "I",
    True: "R",
}
INJECTION_TYPE_CONVERSION = {
    "activation": "o",
    "weight": "w",
}

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
        # ("bit_random", True),
        # ("bit_random", False),
        ("bit_weighting", "random"),
        ("bit_weighting", "linear"),
        ("bit_weighting", "exponential"),
        ("bit_weighting", "gradient"),
        ("approximate_activation_gradient_value", True),
        ("approximate_activation_gradient_value", False),
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
        else:
            extra_injection_info = {}

        self.extra_injection_info = extra_injection_info

        # we remove the batch dimension
        self.summary = torchinfo.summary(model=model, input_size=test_input.size()[1:], batch_dim=0, verbose=0, device="cpu")

        self.random_generator = random.Random(x=seed)
        self.random_threshold = random_threshold

        self.__sensitivity_by_layer_and_index = {}

    @classmethod
    def _init_only_attributions():
        pass

    def load_attributions(self, save_file: os.PathLike, override_settings=True):
        save_file = pathlib.Path(save_file)
        data = torch.load(str(save_file))
        self.summary = data.get("summary", None)
        self.sensitivity_class_name = data.get("sensitivity_class", None)
        self.sensitivity = self.SENSITIVITY_CLASSES[self.sensitivity_class_name]
        self.leaf_layers = data.get("leaf_layers", None)
        self.sensitivity_layers = data.get("sensitivity_layers", None)
        if override_settings:
            self.injection_type = data.get("injection_type", None)
            self.extra_injection_info = data.get("extra_injection_info", None)
            self.random_generator = data.get("random_generator", None)
            self.random_threshold = data.get("random_threshold", None)

    def save_attributions(self, save_file: os.PathLike):
        save_file = pathlib.Path(save_file)
        data = {"injection_type": self.injection_type, "sensitivity_class": self.sensitivity_class_name, "leaf_layers": self.leaf_layers, "sensitivity_layers": self.sensitivity_layers, "extra_injection_info": self.extra_injection_info, "summary": self.summary, "random_threshold": self.random_threshold, "random_generator": self.random_generator}
        torch.save(data, str(save_file))

    def copy_attributions(self, importancesampling):
        self.leaf_layers = importancesampling.leaf_layers
        self.sensitivity_layers = importancesampling.sensitivity_layers

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

    def _generate_bit_index_old(self, random=None):
        # default is to use the value from extra_injection_info
        # if that is None we use the random value passed to the function
        # if also that is None we use importance bit
        if self.extra_injection_info is not None:
            bit_random = self.extra_injection_info.get("bit_random", bool(random) if random is not None else False)
        elif random is not None:
            bit_random = bool(random)
        else:
            bit_random = False

        if bit_random:
            # for random sampling we use a completely random choice
            bit_index = self.random_generator.choices(
                list(range(0, 32)),
            )[0]
        else:
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

        return bit_index

    def _tensor_value_to_torch_bit_list_0_lsb(self, tensor_value, tensor_format):
        if tensor_format == "fp32":
            numpy_value = tensor_value.detach().cpu().numpy()
            numpy_value_integer = numpy_value.view(numpy.dtype(f"u{numpy_value.dtype.itemsize}"))
            binary_string = f"{int(numpy_value_integer):032b}"
            bits_torch_0_lsb = [torch.tensor(float(b), requires_grad=True) for b in binary_string][::-1]
            return bits_torch_0_lsb
        raise ValueError("unknown format")


    def _get_bit_weights_by_gradient(self, tensor_value, tensor_grad, tensor_format):
        bit_list_0_lsb = self._tensor_value_to_torch_bit_list_0_lsb(tensor_value, tensor_format)
        if tensor_format == "fp32":
            sign_bit = bit_list_0_lsb[31]
            exponent_bits = bit_list_0_lsb[23:31]
            mantissa_bits = bit_list_0_lsb[:23]
            # zero check, all bits except sign equal to 0
            if all(map(lambda x: x.item() == 0.0, mantissa_bits + exponent_bits)):
                zero_not_flag = 0
            else:
                zero_not_flag = 1
            # denormalized numbers check, all exponent bits equal to 0
            if all(map(lambda x: x.item() == 0.0, exponent_bits)):
                denormalized_flag = 1
            else:
                denormalized_flag = 0
            # if all exponent bits are 1 and no mantissa bit is 1, we have inf,
            # otherwise we get nan
            if all(map(lambda x: x.item() == 1.0, exponent_bits)):
                if all(map(lambda x: x.item() == 0.0, mantissa_bits)):
                    infinity_flag = 1
                    nan_flag = 0
                else:
                    infinity_flag = 0
                    nan_flag = 1
            else:
                infinity_flag = 0
                nan_flag = 0
            # using -1 ** sign leads to nan grad
            sign = (-2) * sign_bit + 1
            exponent = sum([2 ** i * x for i, x in enumerate(exponent_bits)]) - 2 ** 7 + 1 + denormalized_flag
            mantissa = sum([2 ** (i - 23) * x for i, x in enumerate(mantissa_bits)])
            # print(sign, exponent, mantissa)
            torch_float = sign * 2 ** exponent * ((1 - denormalized_flag) + mantissa) * zero_not_flag
            if infinity_flag or nan_flag:
                torch_float = sys.float_info.max / 2 / len(bit_list_0_lsb) / torch_float * torch_float
            torch_float.grad = torch.tensor(tensor_grad)
            torch_float.backward(retain_graph=True)
            bit_weights = [abs(b.grad.item()) for b in bit_list_0_lsb]
            # print(tensor_value, bit_list_0_lsb, bit_weights)
            return bit_weights
        raise ValueError("unknown format")

    def _get_bit_list(self, tensor_format):
        if tensor_format == "fp32":
            return list(range(0, 32))
        raise ValueError("unknown tensor format type")

    def _generate_bit_index(self, weighting=None, tensor_format=None, tensor_grad=None, tensor_value=None):
        if self.extra_injection_info is not None:
            weighting = self.extra_injection_info.get("bit_weighting", weighting)
            tensor_format = self.extra_injection_info.get("bit_tensor_format", tensor_format)
            tensor_grad = self.extra_injection_info.get("bit_tensor_grad", tensor_grad)
            tensor_value = self.extra_injection_info.get("bit_tensor_value", tensor_value)

        old_weighting = weighting

        if tensor_format is None:
            tensor_format = "fp32"
        if (weighting == "gradient" and (tensor_grad is None or tensor_value is None)) or weighting is None:
            weighting = "random"

        if weighting == "random":
            # for random sampling we use a completely random choice
            bit_index = self.random_generator.choices(
                self._get_bit_list(tensor_format=tensor_format),
            )[0]
            bit_weighting = "random"
        elif weighting == "linear":
            bit_index = self.random_generator.choices(
                self._get_bit_list(tensor_format=tensor_format),
                # linear weighting
                weights=list(range(1, 33)),
            )[0]
            bit_weighting = "linear"
        elif weighting == "exponential":
            bit_index = self.random_generator.choices(
                self._get_bit_list(tensor_format=tensor_format),
                # exponential weighting
                weights=[2 ** x for x in range(0, 32)]
            )[0]
            bit_weighting = "exponential"
        elif weighting == "gradient":
            weights = self._get_bit_weights_by_gradient(tensor_value=tensor_value, tensor_grad=tensor_grad, tensor_format=tensor_format)
            bit_weighting = "gradient"
            if all(map(lambda x: x == 0, weights)):
                if old_weighting == "random":
                    weights = [1] * 32
                    bit_weighting = "random"
                elif old_weighting == "linear":
                    weights = list(range(1, 33))
                    bit_weighting = "linear"
                else:
                    weights = [2 ** x for x in range(0, 32)]
                    bit_weighting = "exponential"
            bit_index = self.random_generator.choices(
                self._get_bit_list(tensor_format=tensor_format),
                # gradient weighting
                weights=weights,
            )[0]
        else:
            raise ValueError("unknown bit weighting method")

        return weighting, bit_index, bit_weighting

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
            random_neuron = True
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
            index_sample = tuple(index)

            weighting = "random"
        else:
            random_neuron = False
            sensitivity = self.sensitivity_layers[layer_sample]
            # print(sensitivity.size())
            # indices_product = itertools.product(*[range(dim) for dim in sensitivity.size()])
            # sensitivity_by_index = []
            # for index in indices_product:
            #     sensitivity_by_index.append({"index": index, "value": sensitivity[index].item()})
            # indices = [x["index"] for x in sensitivity_by_index]
            # we use abs as some attributions might be negative, and we are interested
            # in their magnitudes
            # weights = [abs(x["value"]) for x in sensitivity_by_index]
            if self.__sensitivity_by_layer_and_index.get(layer_sample, None) is None:
                indices_product = itertools.product(*[range(dim) for dim in sensitivity.size()])
                indices = []
                weights = []
                for index in indices_product:
                    indices.append(index)
                    weights.append(abs(sensitivity[index].item()))
                self.__sensitivity_by_layer_and_index[layer_sample] = {"indices": indices, "weights": weights}
            else:
                indices = self.__sensitivity_by_layer_and_index[layer_sample]["indices"]
                weights = self.__sensitivity_by_layer_and_index[layer_sample]["weights"]
            index_sample = tuple(self.random_generator.choices(population=indices, weights=weights, k=1)[0])

            weighting = "gradient"

        if self.injection_type == "weight":
            weight_tensor_value = layer_sample.module.weight[tuple(index_sample)]
            weight_tensor_grad = self.sensitivity_layers[layer_sample][tuple(index_sample)].item()
        elif self.injection_type == "activation" and self.extra_injection_info is not None and self.extra_injection_info.get("approximate_activation_gradient_value", None) is True:
            # we approximate the activation with its gradient, maybe it works ok
            # value must be tensor, grad must be float
            weight_tensor_value = self.sensitivity_layers[layer_sample][tuple(index_sample)]
            weight_tensor_grad = weight_tensor_value.item()
        else:
            weight_tensor_value = None
            weight_tensor_grad = None
        bit_weighting, bit_index, bit_weighting_code = self._generate_bit_index(weighting=weighting, tensor_grad=weight_tensor_grad, tensor_value=weight_tensor_value)

        if self.extra_injection_info is None or not self.extra_injection_info.get("sparse_target", False):
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

        experiment_code = EXPERIMENT_CODE_TEMPLATE.format(BIT_WEIGHTING_CODE_CONVERSION.get(bit_weighting_code.lower(), None), NEURON_IMPORTANCE_CONVERSION.get(random_neuron, None), INJECTION_TYPE_CONVERSION.get(self.injection_type.lower(), None))

        return {"layer": layer_name, "index": index_sample, "bit_index": bit_index, "bit_weighting": bit_weighting, "injection_type": self.injection_type, "sparse_target": sparse_target, "random_neuron": random_neuron, "experiment_code": experiment_code}
