import copy
import functools
import io
import pprint
import pathlib
import random
import sys
import urllib.request

import numpy
import PIL
import pytorch_lightning
import pl_bolts
import torch
import torchvision

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_PARENT_DIR = (CURRENT_DIR / '..').resolve()
DATA_DIR = (CURRENT_DIR / '../data/cifar10_pretrained/').resolve()
DATASET_DIR = (CURRENT_DIR / '../data').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import enpheeph.fi.injection.injectioncallback
import enpheeph.fi.utils.enums.bitvalue
import enpheeph.fi.utils.enums.endianness
import enpheeph.fi.utils.enums.parametertype
import enpheeph.utils.functions

sys.path.append(str(DATA_DIR))

import vgg


### REPRODUCIBILITY
# this flag is used for determinism in PyTorch Lightning Trainer
DETERMINISTIC_FLAG = True
# we call this function to enable reproducibility
enpheeph.utils.functions.enable_determinism(DETERMINISTIC_FLAG)
### REPRODUCIBILITY


class PLModelWrapper(pytorch_lightning.LightningModule):
    def __init__(self, model, normalize_prob_func, loss, accuracy_func):
        super().__init__()

        self.model = model
        self.loss_func = loss
        self.normalize_func = normalize_prob_func
        self.accuracy_func = accuracy_func

    def forward(self, input_):
        return self.model(input_)

    def inference_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.normalize_func(self.model(x))
        loss = self.loss_func(y_hat, y)
        acc = self.accuracy_func(y_hat, y)

        return {'acc': acc, 'loss': loss}

    def validation_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {'val_acc': m['acc'], 'val_loss': m['loss']}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {'test_acc': m['acc'], 'test_loss': m['loss']}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return metrics


# we load the model and define the PL wrapper with it
# we can use VGG11_bn as the model dict is saved in the same directory
# this is trained on CIFAR10
vgg11_bn = vgg.vgg11_bn(pretrained=True)
wrapper = PLModelWrapper(
        vgg11_bn,
        functools.partial(torch.nn.functional.softmax, dim=1),
        torch.nn.functional.cross_entropy,
        pytorch_lightning.metrics.Accuracy(),
        )
# we used the pytorch_lightning CIFAR 10 datamodule
datamodule = pl_bolts.datamodules.CIFAR10DataModule(
        data_dir=str(DATASET_DIR),
        )

# here we define our faults to be tested
faults = []

# this is the weight fault, setting all the weights in the last fully-connected
# layer to zero, covering all the bits
weight_fault = enpheeph.fi.injection.faultdescriptor.FaultDescriptor(
        module_name='model.classifier.0',
        parameter_type=enpheeph.fi.utils.enums.parametertype.ParameterType.DNNWeightDense,
        tensor_index=[0, 0],
        bit_index=[0],
        bit_value=enpheeph.fi.utils.enums.bitvalue.BitValue.StuckAtZero,
        # default parameter name for weight injection
        parameter_name='weight',
        # default endianness, little, so 31 is MSB
        endianness=enpheeph.fi.utils.enums.endianness.Endianness.Little,
)

# here we have the activation fault on the first conv layer output
activation_fault = enpheeph.fi.injection.faultdescriptor.FaultDescriptor(
    module_name='model.features.0',
    parameter_type=enpheeph.fi.utils.enums.parametertype.ParameterType.DNNActivationDense,
    tensor_index=[0, ..., ...],
    bit_index=[0, 10, 32],
    bit_value=enpheeph.fi.utils.enums.bitvalue.BitValue.BitFlip,
    # we don't need any parameter_name, as we use the whole tensor for
    # the output
    # parameter_name=None,
    # default endianness, little, so 31 is MSB
    endianness=enpheeph.fi.utils.enums.endianness.Endianness.Little,
)

faults.append(weight_fault)
faults.append(activation_fault)

callback = enpheeph.fi.injection.injectioncallback.InjectionCallback(
        fault_descriptor_list=faults,
        enabled=False,
        auto_model_init_on_test_start=True,
        auto_load_types=True,
        enabled_faults=[],
)

# not required as we auto init the model on test start
# callback.init_model(wrapper)

trainer = pytorch_lightning.Trainer(
    callbacks=[callback],
    deterministic=DETERMINISTIC_FLAG,
    # with GPU it is very slow due to the memory transfers
    gpus=[0],
    )

print('\n\nBaseline, no injection\n')
# we use this as baseline, no injections
trainer.test(wrapper, datamodule=datamodule)

# we enable the callback now
callback.enable()
# we enable the weight injection
callback.enable_faults([weight_fault])
print('\n\nOnly weight injection\n')

# we test again, only weight injection
trainer.test(wrapper, datamodule=datamodule)

# we enable the activation injection
callback.enable_faults([activation_fault])
print('\n\nWeight + activation injection\n')

# we test again, weight + activation injection
trainer.test(wrapper, datamodule=datamodule)

# we disable the weight injection, only activation
callback.disable_faults([weight_fault])
print('\n\nOnly activation injection\n')

# we test again, activation injection
trainer.test(wrapper, datamodule=datamodule)

# we disable the callback
callback.disable()
print('\n\nBaseline again, no injection\n')

# we test again to reach same results as before injection
trainer.test(wrapper, datamodule=datamodule)

# # here we define the url, we open it and we load it into a temp binary stream
# # in this way it can be opened if offline by PIL
# url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
# temp = io.BytesIO()
# with urllib.request.urlopen(url) as image_url_file:
#     temp.write(image_url_file.read())
#     temp.flush()
#     temp.seek(0)
#     input_image = PIL.Image.open(temp)

# # define the preprocessing pipeline
# preprocess = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(256),
#     torchvision.transforms.CenterCrop(224),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# # preprocess the image
# input_tensor = preprocess(input_image)
# # create a mini-batch as expected by the model
# input_batch = input_tensor.unsqueeze(0)
