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
SRC_PARENT_DIR = (CURRENT_DIR / '..' / '..').resolve()
MODEL_CHECKPOINT = (SRC_PARENT_DIR / "model_training/lightning/snn_mnist_conv/default/12/checkpoints/epoch=3-step=2999.ckpt").resolve()
DATASET_DIR = (SRC_PARENT_DIR / 'data/mnist').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import src.fi.injection.injectioncallback
import src.fi.utils.enums.bitvalue
import src.fi.utils.enums.endianness
import src.fi.utils.enums.parametertype
#import src.ml.models.snnwrapper
import src.ml.pl.models.vision.plvisionwrapper
import src.utils.functions


### REPRODUCIBILITY
# this flag is used for determinism in PyTorch Lightning Trainer
DETERMINISTIC_FLAG = True
# we call this function to enable reproducibility
src.utils.functions.enable_determinism(DETERMINISTIC_FLAG)
### REPRODUCIBILITY


# we load the model and define the PL wrapper with it
# we can use VGG11_bn as the model dict is saved in the same directory
# this is trained on CIFAR10
model = src.ml.pl.models.vision.plvisionwrapper.PLVisionWrapper.load_from_checkpoint(str(MODEL_CHECKPOINT))
# we used the pytorch_lightning CIFAR 10 datamodule
datamodule = pl_bolts.datamodules.MNISTDataModule(
        data_dir=str(DATASET_DIR),
        )

# here we define our faults to be tested
faults = []

voltage_state_fault = src.fi.injection.faultdescriptor.FaultDescriptor(
        module_name='model.model.1',
        parameter_type=src.fi.utils.enums.parametertype.ParameterType.SNNStateLIFStateVoltageDense,
        tensor_index=...,
        bit_index=...,
        bit_value=src.fi.utils.enums.bitvalue.BitValue.BitFlip,
        # default endianness, little, so 31 is MSB
        endianness=src.fi.utils.enums.endianness.Endianness.Little,
)

faults.append(voltage_state_fault)

callback = src.fi.injection.injectioncallback.InjectionCallback(
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
    # to use the trained model
    resume_from_checkpoint=str(MODEL_CHECKPOINT)
    )

print('\n\nBaseline, no injection\n')
# we use this as baseline, no injections
trainer.test(model, datamodule=datamodule)

# we enable the callback now
callback.enable()
# we enable the weight injection
callback.enable_faults([voltage_state_fault])
print('\n\nVoltage State injection\n')

# we test again, weight + activation injection
trainer.test(model, datamodule=datamodule)

# we disable the callback
callback.disable()
print('\n\nBaseline again, no injection\n')

# we test again to reach same results as before injection
trainer.test(model, datamodule=datamodule)

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
