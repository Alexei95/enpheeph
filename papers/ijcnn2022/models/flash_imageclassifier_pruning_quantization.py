# -*- coding: utf-8 -*-
import pathlib

import flash
import flash.core.data.utils
import flash.image


CURRENT_DIRECTORY = pathlib.Path(__file__).absolute().parent
DATASET_DIRECTORY = pathlib.Path("/shared/ml/flash/datasets/")
if not DATASET_DIRECTORY.exists():
    DATASET_DIRECTORY = CURRENT_DIRECTORY / "data" / "flash" / "datasets"
    DATASET_DIRECTORY.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIRECTORY = pathlib.Path("/shared/ml/flash/checkpoints/")
if not CHECKPOINT_DIRECTORY.exists():
    CHECKPOINT_DIRECTORY = CURRENT_DIRECTORY / "data" / "flash" / "checkpoints"
    CHECKPOINT_DIRECTORY.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "classification_dataset"
DATASET_URL = "https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip"

# 1. Create the DataModule
flash.core.data.utils.download_data(DATASET_URL, str(DATASET_DIRECTORY / DATASET_NAME))

datamodule = flash.image.ImageClassificationData.from_folders(
    train_folder=str(DATASET_DIRECTORY / DATASET_NAME / "hymenoptera_data/train/"),
    val_folder=str(DATASET_DIRECTORY / DATASET_NAME / "hymenoptera_data/val/"),
    test_folder=str(DATASET_DIRECTORY / DATASET_NAME / "hymenoptera_data/test/"),
    batch_size=64,
)

# 2. Build the task
model = flash.image.ImageClassifier(
    backbone="resnet18", num_classes=datamodule.num_classes
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=[2])
trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")

# test using the test_folder from datamodule
predictions = trainer.test(model, datamodule)

# 5. Save the model!
trainer.save_checkpoint(
    CHECKPOINT_DIRECTORY / (DATASET_NAME + "image_classification_model.pt")
)
