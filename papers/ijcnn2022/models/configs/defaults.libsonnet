{
    "batch_size": 32,
    "dataset_root_dir": "/shared/ml/datasets/vision/",
    "gpu_devices": [
        2,
    ],
    "num_workers": 64,
    "root_dir": "model_training/lightning",
    "seed_everything": 42,
    # problems with 0.2 as it becomes 0.20...01
    "val_split": 0.21,
}
