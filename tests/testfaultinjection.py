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

import src.fi.baseinjectioncallback
import src.utils

sys.path.append(str(DATA_DIR))

import vgg


### REPRODUCIBILITY
# this flag is used for determinism in PyTorch Lightning Trainer
DETERMINISTIC_FLAG = True
# we call this function to enable reproducibility
src.utils.enable_determinism(DETERMINISTIC_FLAG)
### REPRODUCIBILITY


class PLWrapper(pytorch_lightning.LightningModule):
    def __init__(self, model, normalize_prob_func, loss):
        super().__init__()

        self.model = model
        self.loss_func = loss
        self.normalize_func = normalize_prob_func

    def forward(self, input_):
        return self.model(input_)

    def inference_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.normalize_func(self.model(x))
        loss = self.loss_func(y_hat, y)
        acc = pytorch_lightning.metrics.functional.accuracy(y_hat, y)

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
wrapper = PLWrapper(
        vgg11_bn,
        functools.partial(torch.nn.functional.softmax, dim=1),
        torch.nn.functional.cross_entropy)
datamodule = pl_bolts.datamodules.CIFAR10DataModule(
        data_dir=str(DATASET_DIR),
        )

faults = [
    src.fi.basefaultdescriptor.BaseFaultDescriptor(
        module_name='model.classifier.6',
        parameter_type=src.fi.basefaultdescriptor.ParameterType.Weight,
        tensor_index=...,
        bit_index=slice(0, 30),
        bit_value=src.fi.basefaultdescriptor.BitValue.StuckAtZero,
        # default parameter name for weight injection
        parameter_name='weight',
    ),
]
callback = src.fi.baseinjectioncallback.BaseInjectionCallback(
        fault_descriptor_list=faults,
        enabled=False,
        auto_model_init_on_test_start=True,
        )

# not required as we auto init the model on test start
# callback.init_model(wrapper)

trainer = pytorch_lightning.Trainer(
    callbacks=[callback],
    deterministic=DETERMINISTIC_FLAG,
    gpus=[0],
    )

# we use this as baseline
trainer.test(wrapper, datamodule=datamodule)

# we enable the callback now
callback.enable()

# we test again
trainer.test(wrapper, datamodule=datamodule)

# we disable the callback
callback.disable()

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