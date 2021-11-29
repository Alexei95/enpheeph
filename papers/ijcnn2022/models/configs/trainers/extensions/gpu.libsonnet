function(
    devices,
    trainer_config,
)
trainer_config + {
    "trainer"+: {
        "accelerator": "gpu",
        "devices": devices,
        "move_metrics_to_cpu": false,
    },
}
