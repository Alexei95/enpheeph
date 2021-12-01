function(enable_qat=true)
    if enable_qat then
        {
            local config = self,

            quantization_at_enabled:: true,

            quantization_at_input_compatible:: true,
            quantization_at_qconfig:: "fbgemm",
            quantization_at_quantize_on_fit_end:: false,
            # we use average instead of histogram as histc is non-deterministic
            quantization_at_observer:: "average",
            quantization_at_observer_stages+:: {},
            quantization_at_modules_to_fuse+:: {},

            trainer_callbacks+:: {
                "quantization_aware_training"+: {
                    "class_path":
                        "pytorch_lightning.callbacks.QuantizationAwareTraining",
                    "init_args"+: {
                        # (Union[Callable, int, None]) – count or custom function
                        # to collect quantization statistics:
                        # None (deafult). The quantization observer is called in
                        # each module forward
                        # (useful for collecting extended statistic
                        # when useing image/data augmentation).
                        # int. Use to set a fixed number of calls,
                        # starting from the beginning.
                        # Callable. Custom function with single trainer argument.
                        # See this example to trigger only the last epoch:
                        # def custom_trigger_last(trainer):
                        #     return trainer.current_epoch == (trainer.max_epochs - 1)
                        # QuantizationAwareTraining
                        # (collect_quantization=custom_trigger_last)
                        "collect_quantization": null,

                        # (bool) – preserve quant/dequant layers.
                        # This allows to feat any input as to the original model,
                        # but break compatibility to torchscript
                        # and export with torch.save.
                        "input_compatible": config.quantization_at_input_compatible,

                        # (Optional[Sequence]) – allows you fuse a few layers together
                        # as shown in diagram to find which
                        # layer types can be fused, check
                        # https://github.com/pytorch/pytorch/pull/43286.
                        "modules_to_fuse":
                            if std.length(
                                config.quantization_at_modules_to_fuse
                            ) > 0 then
                                std.objectValues(
                                    config.quantization_at_modules_to_fuse
                                ),

                        # (Sequence[str]) – allow fake-quantization modules’ observers
                        # to do calibration during provided stages:
                        # 'train': the observers can do calibration during training.
                        # 'validate': the observers can do calibration
                        # during validating.
                        # Note that we don’t disable observers during the sanity check
                        # as the model hasn’t been calibrated with training data yet.
                        # After the sanity check, the fake-quantization modules
                        # are restored to initial states.
                        # 'test': the observers can do calibration during testing.
                        # 'predict': the observers can do calibration during predicting.
                        # Note that we only handle observers belonging to
                        # fake-quantization modules.
                        # When qconfig is a str and observer_type is 'histogram',
                        # the observers won’t belong to any fake-quantization modules
                        # and will not be controlled by the callback.
                        "observer_enabled_stages"+: std.objectValues(
                            config.quantization_at_observer_stages
                        ),

                        # (str) – allows switching between
                        # MovingAverageMinMaxObserver as “average”
                        # (default) and HistogramObserver as “histogram”
                        # which is more computationally expensive.
                        "observer_type": config.quantization_at_observer,

                        # (Union[str, QConfig]) – quantization configuration:
                        # ’fbgemm’ for server inference.
                        # ’qnnpack’ for mobile inference.
                        # a custom torch.quantization.QConfig.
                        "qconfig": config.quantization_at_qconfig,

                        # (bool) – perform the quantization in on_fit_end.
                        # Note that once converted,
                        # the model cannot be put in training mode again.
                        "quantize_on_fit_end": config.quantization_at_quantize_on_fit_end,
                    },
                },
            },
        }
    else
        {quantization_at_enabled:: false}
