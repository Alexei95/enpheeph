# enpheeph - configuration files

These configuration files contain the definitions for the different models and datasets to be used during the experiments.

## Structure

The overall structure of the files is as follows:

1. ``base.libsonnet`` is the basic library which defines the default variables, e.g. seed value, from the ``defaults.libsonnet`` definitions
2. ``defaults.libsonnet`` contains some defaults for seed and other variables so that they can be easily changed if needed, without having to look for the variables in across the different files
3. ``utils.libsonnet`` contains the function definitions for useful utilities, such as path joining and class attribute access
4. ``datasets`` contains all the definitions for the dataset instantiation
   1. ``base_dataset.libsonnet`` contains the base definitions needed for the different datasets, like batch size, path, number of workers, ...
   2. ``select_dataset.libsonnet`` is a helper function which selects the specific dataset definition based on its name
      1. The downside of this approach is that all the dataset definitions must be loaded in the global mapping ``name -> definition``, so this might exponentially worsen if there are too many datasets
      2. Each dataset must also be added manually to the list, hence the reason why only a selected subset of datasets is chosen
   3. Each remaining file contains the specifics for each dataset with a detailed descriptions, so they can be updated if needed
5. ``experiments`` contains the final ``.json`` output files which are used with the PyTorch Lightning Trainer
   1. Additionally it contans the ``base_task.jsonnet`` which is the file to be compiled with all the proper settings in place to generate the proper output
6. ``models`` contains the definitions for the different models to be used
   1. Also here we have ``select_model.libsonnet`` which allows for the selection of the different models with minimal changes in the external code, as it is the same as ``select_dataset.libsonnet``
   2. ``base_model.libsonnet`` defines the basic defaults for the models: this is done following the models available in PyTorch Lightning Flash, so that the variables are in common across different models
7. ``trainers`` contains all the settings for the PyTorch Lightning Trainer, together with the different device extensions and the custom training flags like pruning and quantization-aware training
   1. The different extensions are located in ``extensions``, e.g. ``gpu.libsonnet`` contains the GPU extensions, as by default the training happens on the CPU
   2. For the other settings, we use the ``callbacks`` folder, as all these algorithms modify the inner workings of the Trainer, so they need to be called during the training loop
   3. ``flash_trainer.libsonnet`` and ``pytorch_lightning_trainer.libsonnet`` define the basic flags for the PyTorch Lightning Flash Trainer and the basic PyTorch Lightning Trainer
      1. They need to be called in order (``pytorch_lightning_trainer.libsonnet`` and then ``flash_trainer.libsonnet``) as the variables overlap in part.

## How-To

```bash
jsonnet \
--ext-str dataset_name=carla --ext-code enable_pruning=true --ext-code enable_qat=false --ext-str model_backbone=mobilenetv3_large_100 --ext-str model_head=fpn --ext-str model_task=semantic_segmentation --ext-str dataset_root=<dataset_root_path> --ext-str dataset_batch_size=16 \
<input_file_path> -o <output_file_path>
```

1. ``--ext-str`` pass the following ``variable_name=variable_str`` as external string variable inside the JSONNet file
