local defaults = import "../../defaults.libsonnet";

{
    local config = self,
    trainer_gpu_devices+:: defaults.trainer_gpu_devices,

    "trainer"+: {
        "accelerator": "gpu",
        # we cannot join with +: as the default is null in trainer,
        # so it cannot be join-overridden
        "devices": std.objectValues(config.trainer_gpu_devices),
        "move_metrics_to_cpu": false,
    },
}
