{
    "seed_everything": 42,
    "dataset_batch_size": 32,
    "dataset_num_workers": 64,
    "dataset_root": "/shared/ml/datasets/vision/",
    "dataset_val_split": 0.20,
    "trainer_gpu_devices": {
        "2": 2,
    },
    "trainer_monitor_metric": "val_accuracy",
    "trainer_name": "pl-flash-cli",
    "trainer_root_dir": "model_training/lightning",
}
