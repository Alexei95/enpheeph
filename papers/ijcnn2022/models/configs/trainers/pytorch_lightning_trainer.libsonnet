# we define the config as a function,
# so that we require the arguments to be passed to work properly
function(
    name,
    root_dir,
    seed_everything,
)
{
    # we save hidden fields to be used by child config
    name:: name,
    root_dir:: root_dir,
    complete_dir:: $['root_dir'] + '/' + $['name'],

    ## this is the normal configuration
    # to set the seed for Python's random, numpy.random and torch.random
    # Set to an int to run seed_everything with this value before classes instantiation
    # (type: Optional[int], default: null)
    "seed_everything": seed_everything,

    ## NOTE: this is the custom configuration for the Trainer class
    # it is defined here so that it can be used in common across all the models
    "trainer": {

        # to select a distributed backend accelerator
        # since we train on a single system with multiple GPU, we can use dp
        # however this may lead to memory bottleneck, as it would move parts of each
        # batch to different GPUs
        # so to use each GPU independently but sync gradients we use ddp
        # default is None, to determine automatically, or it can accept a custom
        # accelerator object
        # if some options need to be changed just switch this to None and
        # add the pytorch_lightning.plugins.DDPPlugin to the plugins list
        # NOTE: we are switching this to None to allow for find_unused_parameters to
        # be False, so that we gain in performance
        # Supports passing different accelerator types
        # ("cpu", "gpu", "tpu", "ipu", "auto")
        # as well as custom accelerator instances.
        # .. deprecated:: v1.5
        # Passing training strategies (e.g., 'ddp') to ``accelerator``
        # has been deprecated in v1.5.0
        # and will be removed in v1.7.0.
        # Please use the ``strategy`` argument instead.
        # (type: Union[str, Accelerator, null], default: null)
        ##### "accelerator": null,

        # number of batches to accumulate before propagating the gradients
        # in this way we multiply the effective batch size by this number
        # it can be also a dict mentioning the epoch at which a new accumulation
        # number starts
        # Accumulates grads every k batches or as set up in the dict.
        # (type: Union[int, Dict[int, int], null], default: null)
        "accumulate_grad_batches": null,

        # to select the backend for limited-precision training
        # native is torch.cuda.amp, apex for NVIDIA apex
        # the backend cannot be disabled here, but it must be changed with the
        # options
        # default is native
        # The mixed precision backend to use ("native" or "apex").
        # (type: str, default: native)
        "amp_backend": "native",

        # to select the type of limited-precision training
        # O0 is FP-32 training
        # O1 is mixed-precision, with internal whitelist/blacklist
        # it modifies most operations to be done on FP16, except where additional
        # precision may be helpful, e.g. softmax
        # O2 instead modifies the weights and the activations but not the functions
        # having lower precision while running but updates are done on FP32
        # O3 does both O1 and O2, thus running only on FP16, and may be unstable but
        # it is the fastest available
        # default is O2, but in our case we want FP32 training, which is O0
        # The optimization level to use (O1, O2, etc...).
        # By default it will be set to "O2"
        # if ``amp_backend`` is set to "apex".
        # (type: Optional[str], default: null)
        "amp_level": "O0",

        # to find the optimal learning rate, if True it is saved in
        # hparams.learning_rate, otherwise it is saved in hparams.<string> if it is
        # a string, default False
        # the model or the datamodule must contain the name used for the parameter
        # NOTE: this passage is not done if we don't call Trainer.tune(), as it
        # happens with the LightningCli in PyTorch Lightning v1.3.2
        # If set to True, will make trainer.tune() run a learning rate finder,
        # trying to optimize initial learning for faster convergence.
        # trainer.tune() method will
        # set the suggested learning rate in self.lr or self.learning_rate
        # in the LightningModule.
        # To use a different key set a string instead of True with the key name.
        # (type: Union[bool, str], default: False)
        "auto_lr_find": false,

        # to autoscale the batch size to the optimal depending on the memory
        # if None it is not done, otherwise it must be an algorithm like binsearch
        # default None, but actually None is not allowed, only False
        # the result is saved in hparams.batch_size
        # the model or the datamodule must contain the name used for the parameter
        # NOTE: this passage is not done if we don't call Trainer.tune(), as it
        # happens with the LightningCli in PyTorch Lightning v1.3.2
        # If set to True, will `initially` run a batch size
        # finder trying to find the largest batch size that fits into memory.
        # The result will be stored in self.batch_size in the LightningModule.
        # Additionally, can be set to either `power` that estimates
        # the batch size through
        # a power search or `binsearch` that estimates the batch size
        # through a binary search.
        # (type: Union[str, bool], default: False)
        "auto_scale_batch_size": false,

        # to automatically choose the unoccupied GPUs
        # If enabled and ``gpus`` is an integer, pick available
        # gpus automatically. This is especially useful when
        # GPUs are configured to be in "exclusive mode", such
        # that only one process at a time can access them.
        # (type: bool, default: False)
        "auto_select_gpus": false,

        # this flag enables cudnn.benchmark to find the best algorithm for the system
        # but it is static, i.e. it does not check all the possible input sizes, so
        # it may make the system slower
        # default is False
        # If true enables cudnn.benchmark. (type: bool, default: False)
        "benchmark": false,

        # since PyTorch Lightning 1.5, instances can be generated in callbacks directly
        # not needing the direct support anymore
        # Add a callback or list of callbacks.
        # (type: Union[List[Callback], Callback, null], default: null)
        "callbacks": [
            # this callback implements early stopping, that is stopping the training
            # once a monitored validation metric has not improved for a certain number
            # of epochs
            {
                "class_path": "pytorch_lightning.callbacks.EarlyStopping",
                "init_args": {
                    # if True it checks the monitored metric to be finite, stopping when
                    # it becomes NaN or infinite
                    # default is True
                    "check_finite": true,
                    # if True the monitored metric is compared at the end
                    # of the training epoch,
                    # if False it is checked after the validation phase
                    # default is False, to check after validation
                    "check_on_train_epoch_end": false,
                    # if the monitored metric becomes worse than this threshold the
                    # training is stopped
                    # default is None to disable it
                    "divergence_threshold": null,
                    # minimum difference to consider as improvement, an absolute change
                    # smaller than delta will not be considered as an improvement
                    # default is 0.0, all improvements are considered
                    "min_delta": 0.001,
                    # min minimizes the target, max maximizes it
                    # default is min, since we are usign a loss we want to minimize it
                    "mode": "min",
                    # string of monitored metric
                    # default is early_stop_on
                    "monitor": "val_loss_epoch",
                    # number of epochs to continue running without improvements before
                    # stopping
                    # default is 3
                    "patience": 5,
                    # if the monitored metric reaches this threshold, training is stopped
                    # default is None to disable it
                    "stopping_threshold": null,
                    # if True it crashes if the monitored metric is not found
                    # default is True
                    "strict": true,
                    # if True we print more info about it
                    # default is False
                    "verbose": true,
                },
            },

            # this callback logs the stats of the GPU, it must be disabled if running
            # without one, otherwise it will block the execution
            # NOTE: it may slow down execution as it uses nvidia-smi output
            {
                "class_path": "pytorch_lightning.callbacks.GPUStatsMonitor",
                "init_args": {
                    # each of the following flags enable
                    # the corresponding resource logging
                    "fan_speed": true,
                    "gpu_utilization": true,
                    "intra_step_time": true,
                    "inter_step_time": true,
                    "memory_utilization": true,
                    "temperature": true,
                },
            },

            # this callback monitors the learning rate of the model
            # it is useful for dynamic learning rates, a bit useless if the learning
            # rate is constant
            {
                "class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
                "init_args": {
                    # whether to log also the momentum of the learning rate
                    "log_momentum": true,
                    # whether to log the learning rate every epoch or every step
                    "logging_interval": "epoch",
                },
            },

            # this callback can be customized to checkpoint the trained model
            # depending on different parameters
            {
                "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                "init_args": {
                    # where the models will be saved, by default if None it relates to
                    # the Trainer default_root_dir
                    # when it is None it will use the logger directory (if there is only
                    # one) as saving path, creating a sub-directory checkpoints
                    "dirpath": null,
                    # saves a checkpoint every n validation epochs
                    "every_n_epochs": 1,
                    # number of training steps between checkpoints
                    # if 0 or None it is skipped
                    "every_n_train_steps": null,
                    # the filename to use for saving the checkpoints
                    # it can use variable expansion
                    "filename": null,
                    # whether to min(imize) or max(imize) the monitoreq quantity
                    "mode": "min",
                    # this indicates the quantity to monitor
                    # if None it saves only the last model
                    "monitor": "val_loss_epoch",
                    # to always save the last model
                    "save_last": true,
                    # to keep the top-k models depending on the monitored quantity
                    # if 0 none is saved, if -1 all of them, if multiple savings per epoch
                    # each call will have a v1, v2, ... appended
                    "save_top_k": 3,
                    # whether to save only the weights
                    "save_weights_only": false,
                    # to toggle verbosity mode
                    "verbose": true,
                },
            },

            # this callback prints the results of the training on the stdout,
            # using tqdm
            {
                "class_path": "pytorch_lightning.callbacks.TQDMProgressBar",
                "init_args": {
                    # number of lines to displace the counter, so that other counters
                    # can be shown
                    "process_position": 0,
                    # number of batches between updates
                    "refresh_rate": 1,
                },
            },
        ],

        # this is the default checkpointing callback, which is enabled if no custom
        # checkpointing is used
        # it must be True if we use checkpointing
        # NOTE: it is unsupported since PyTorch Lightning v1.3
        # deprecated, see "enable_checkpointing"
        ##### "checkpoint_callback": true,

        # how many training epochs pass in-between runs of validation
        # default is 1
        # Check val every n train epochs. (type: int, default: 1)
        "check_val_every_n_epoch": 1,

        # the default root dir for checkpoints and logging, with respect to the
        # project root
        # we use a handle to copy it in the other directory settings
        # default is os.getcwd
        # Default path for logs and weights when no logger/ckpt_callback passed.
        # Default: ``os.getcwd()``.
        # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
        # (type: Optional[str], default: null)
        "default_root_dir": $['complete_dir'],

        # Enable anomaly detection for the autograd engine. (type: bool, default: False)
        # **IMPORTANT**: enable only for debuggin as it enables extra tests and
        # tracebacks
        # https://pytorch.org/docs/stable/autograd.html#anomaly-detection
        "detect_anomaly": false,

        # this flag forces the reproducibility flags to be turned on if True, but
        # it may make the system slower
        # default is False, but we want to ensure reproducibility so we set it to
        # True
        # If ``True``, sets whether PyTorch operations
        # must use deterministic algorithms.
        # Default: ``False``.
        # (type: bool, default: False)
        "deterministic": true,

        # Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
        # based on the accelerator type.
        # (type: Union[int, str, List[int], null], default: null)
        # we use 2 to run on gpu2, the third GPU
        # if null it will choose automatically
        "devices": [
            2,
        ],

        # set the distributed backed
        # NOTE: it has been superseded by accelerator
        "distributed_backend": null,

        # If ``True``, enable checkpointing.
        # It will configure a default ModelCheckpoint callback
        # if there is no user-defined ModelCheckpoint in
        # :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
        # (type: bool, default: True)
        "enable_checkpointing": true,

        # Whether to enable model summarization by default. (type: bool, default: True)
        "enable_model_summary": true,

        # Whether to enable to progress bar by default. (type: bool, default: True)
        "enable_progress_bar": true,

        # this sets the number of runs to do to check that everything is working
        # properly, as it disables callbacks and logging
        # it must be disabled in production, can be set to any integer > 0 in
        # development
        # default is False
        # Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
        # of train, val and test to find any bugs (ie: a sort of unit test).
        # (type: Union[int, bool], default: False)
        "fast_dev_run": false,

        # how many training steps to flush the logs to disk
        # How often to flush logs to disk (defaults to every 100 steps).
        # .. deprecated:: v1.5
        #     ``flush_logs_every_n_steps`` has been deprecated in v1.5
        # and will be removed in v1.7.
        # Please configure flushing directly in the logger instead.
        # (type: Optional[int], default: null)
        ##### "flush_logs_every_n_steps": null,

        # GPUs to be used
        # either int as number, list for selection, -1 to select all the available
        # ones
        # if None it runs on CPU, default is None
        # Number of GPUs to train on (int) or which GPUs to train on
        # (list or str) applied per node
        # (type: Union[int, str, List[int], null], default: null)
        # it is now being deprecated for "devices"
        ##### "gpus": null,

        # value to be used to clip gradients
        # if 0.0, the default, they are not clipped
        # **IMPORTANT**: disabled since we do not need,
        # otherwise it is converted to integer and creates issues
        # The value at which to clip gradients.
        # Passing ``gradient_clip_val=None`` disables
        # gradient clipping. If using Automatic Mixed Precision (AMP),
        # the gradients will be unscaled before.
        # (type: Union[int, float, null], default: null)
        "gradient_clip_val": null,

        # which algorithm to use for gradient clipping, value for clip_by_value,
        # norm for clip_by_norm
        # default is norm
        # The gradient clipping algorithm to use. Pass
        # ``gradient_clip_algorithm="value"``
        # to clip by value, and ``gradient_clip_algorithm="norm"``
        # to clip by norm. By default it will
        # be set to ``"norm"``.
        # (type: Optional[str], default: null)
        "gradient_clip_algorithm": null,

        # How many IPUs to train on.
        # (type: Optional[int], default: null)
        "ipus": null,

        # to use only a percentage of the prediction set, float for percentage and
        # int for number of batches, default is 1.0
        # How much of prediction dataset to check
        # (float = fraction, int = num_batches).
        # (type: Union[int, float], default: 1.0)
        "limit_predict_batches": 1.0,

        # to use only a percentage of the training set, useful for debugging the
        # training loop, if int is the number of batches, if float is the ratio
        # default is 1.0
        # How much of training dataset to check (float = fraction, int = num_batches).
        # (type: Union[int, float], default: 1.0)
        "limit_train_batches": 1.0,

        # limit test and validation datasets, the same as for training batches
        # if multiple dataloaders are used, the limit is intended for each one of
        # them separately
        # How much of test dataset to check (float = fraction, int = num_batches).
        # (type: Union[int, float], default: 1.0)
        "limit_test_batches": 1.0,

        # How much of validation dataset to check
        # (float = fraction, int = num_batches).
        #(type: Union[int, float], default: 1.0)
        "limit_val_batches": 1.0,

        # a true flag enables the default TensorboardLogger,
        # for logging results
        # we pass our custom logger here
        # to pass a custom object, use class_path to describe the class
        # location
        # and init_args to create a dict of the init arguments to
        # pass to the class
        # Logger (or iterable collection of loggers) for experiment tracking.
        # A ``True`` value uses
        # the default ``TensorBoardLogger``. ``False`` will disable logging.
        # If multiple loggers are
        # provided and the `save_dir` property of that logger is not set,
        # local files (checkpoints,
        # profiler traces, etc.) are saved in ``default_root_dir``
        # rather than in the ``log_dir`` of any
        # of the individual loggers.
        # (type: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool],
        # default: True)
        "logger": {
            # this is the default logger included in pytorch-lightning,
            # check the docs
            # for different loggers
            "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
            "init_args": {
                # the main directory for loggers
                "save_dir": $['complete_dir'],
                # experiment name, in this custom configuration it is default
                "name": "default",
                # version of the experiment, if not assigned it is increasing
                # automatically based on the logging directory
                # if a string it is used, otherwise 'version_{$version}'
                # is used
                "version": null,
                # this enables the saving of the computational graph
                # it requires example_input_array in the model
                "log_graph": true,
                # enables a placeholder for log_hyperparams metrics,
                # if none is provided
                "default_hp_metric": true,
                # prefix for the metrics
                "prefix": "",
            }
        },

        # how often a new logging row should be added, without writing to disk
        # directly
        # default is 50
        # How often to log within steps (defaults to every 50 steps).
        # (type: int, default: 50)
        "log_every_n_steps": 10,

        # to log the memory usage on GPU
        # None to disable, min_max to get the limit or all to monitor it
        # its usage is limited on the master node only, and it uses nvidia-smi, so it
        # may slow performance
        # default is None
        # we disable it as we are using the custom callback
        # None, 'min_max', 'all'. Might slow performance.
        # .. deprecated:: v1.5
        #     Deprecated in v1.5.0 and will be removed in v1.7.0
        #     Please use the ``DeviceStatsMonitor`` callback directly instead.
        # (type: Optional[str], default: null)
        ##### "log_gpu_memory": null,

        # number of epochs to run, default is 1000, we use 100 here
        # Stop training once this number of epochs is reached.
        # Disabled by default (None).
        # If both max_epochs and max_steps are not specified,
        # defaults to ``max_epochs = 1000``.
        # To enable infinite training, set ``max_epochs = -1``.
        # (type: Optional[int], default: 3)
        "max_epochs": -1,

        # maximum number of training steps to execute, -1 as default to disable it
        # it will stop the training depending on the earliest occurrence of
        # steps/epochs
        # Stop training after this number of steps. Disabled by default (-1).
        # If ``max_steps = -1``
        # and ``max_epochs = None``, will default to ``max_epochs = 1000``.
        # To enable infinite training, set
        # ``max_epochs`` to ``-1``.
        # (type: int, default: -1)
        "max_steps": -1,

        # maximum number of time to run, stopping mid-epoch
        # it takes precedence over max steps/epochs, and it can be written as dict,
        # i.e. {"days": 1, "hours": 5} or "01:05:00:00"
        # default is None to disable it, and the minimum requirements always take
        # precedence
        # Stop training after this amount of time has passed.
        # Disabled by default (None).
        # The time duration can be specified in the format
        # DD:HH:MM:SS (days, hours, minutes seconds), as a
        # :class:`datetime.timedelta`,
        # or a dictionary with keys that will be passed to
        # :class:`datetime.timedelta`.
        # (type: Union[str, timedelta, Dict[str, int], null], default: null)
        "max_time": null,

        # minimum number of epochs to execute, default is 1
        # Force training for at least these many epochs. Disabled by default (None).
        # If both min_epochs and min_steps are not specified, defaults to
        # ``min_epochs = 1``.
        # (type: Optional[int], default: null)
        "min_epochs": 1,

        # minimum number of training steps to execute, default is None to disable it
        # it will stop at the latest between min steps/epochs
        # Force training for at least these number of steps.
        # Disabled by default (None).
        # (type: Optional[int], default: null)
        "min_steps": null,

        # if True the metrics are done on CPU, which implies memory transfers but
        # also a lower GPU/TPU memory usage
        # default is False
        # Whether to force internal logged metrics to be moved to cpu.
        # This can save some gpu memory, but can make training slower.
        # Use with attention.
        # (type: bool, default: False)
        "move_metrics_to_cpu": false,

        # max_size_cycle ends training with the longest dataloader, and the smaller
        # ones are reloaded when they end
        # min_size instead stops the training with the shortest one, reloading all
        # of them
        # this is useful only with multiple dataloaders
        # default is max_size_cycle, but we prefer min_size
        # How to loop over the datasets when there are multiple train loaders.
        # In 'max_size_cycle' mode, the trainer ends one epoch when the
        # largest dataset is traversed,
        # and smaller datasets reload when running out of their data.
        # In 'min_size' mode, all the datasets
        # reload when reaching the minimum length of datasets.
        # (type: str, default: max_size_cycle)
        "multiple_trainloader_mode": "min_size",

        # number of nodes to use for training, default is 1
        # in our case we have only 1 node, so 1 is fine
        # Number of GPU nodes for distributed training. (type: int, default: 1)
        "num_nodes": 1,

        # number of processes to use, useful when using ddp_cpu or ddp, however
        # it does not provide any speed-up compared to single process when running
        # on CPU only, as multi-threading by PyTorch is already optimized
        # it is set to the number of GPUs when using ddp
        # default is 1, or the number of GPUs for ddp
        # Number of processes for distributed training with ``accelerator="cpu"``.
        # (type: int, default: 1)
        "num_processes": 1,

        # number of validation steps to run before starting the training, to check
        # whether everything is working
        # default is 2, it can be disabled with 0, or check the whole validation
        # dataloader with -1
        # if the check is run, the dataloader is reset before starting the validation
        # we use the default of 2
        # Sanity check runs n validation batches before starting the training routine.
        # Set it to `-1` to run all batches in all validation dataloaders.
        # (type: int, default: 2)
        "num_sanity_val_steps": 2,

        # how many batches to use to overfit the model
        # it will train on a set if float or number of batches if int and use the
        # rest of the training set as validation and test sets, to allow overfitting
        # it disables shuffling when enabled
        # default is 0.0, to disable it
        # we use 0.0 as we don't need to test overfitting
        # **IMPORTANT**: disabled since we do not need,
        # otherwise it is converted to integer and creates issues
        # Overfit a fraction of training data (float) or a set number of batches (int).
        # (type: Union[int, float], default: 0.0)
        "overfit_batches": 0.0,

        # Plugins allow modification of core behavior like ddp and amp,
        # and enable custom lightning plugins.
        # (type: Union[TrainingTypePlugin, PrecisionPlugin, ClusterEnvironment,
        # CheckpointIO, str, List[Union[TrainingTypePlugin, PrecisionPlugin,
        # ClusterEnvironment, CheckpointIO, str]], null], default: null)
        "plugins": null,

        # if set to True it calls prepare_data on the first CPU process/GPU/TPU
        # of each node, set with LOCAL_RANK=0, if False it runs only once on the
        # master node, NODE_RANK=0 and LOCAL_RANK=0
        # default is True, we can use it as we train on a single node
        # If True, each LOCAL_RANK=0 will call prepare data.
        # Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
        # .. deprecated:: v1.5
        # Deprecated in v1.5.0 and will be removed in v1.7.0
        # Please set ``prepare_data_per_node`` in LightningDataModule or
        # LightningModule directly instead.
        # (type: Optional[bool], default: null)
        ##### "prepare_data_per_node": null,

        # precision to use for training, 64/32 for CPU/GPU/TPU, also 16 on GPU/TPU
        # 16 on TPU uses torch.bfloat16, but it shows torch.float32
        # default is 32, we can use this default unless a better precision is
        # required
        # Double precision (64), full precision (32), half precision (16)
        # or bfloat16 precision (bf16).
        # Can be used on CPU, GPU or TPUs.
        # (type: Union[int, str], default: 32)
        "precision": 32,

        # moves the lines of the integrated progress bar, ignored if a custom
        # callback is used
        # default is 0
        # Orders the progress bar when running multiple models on same machine.
        # .. deprecated:: v1.5
        # ``process_position`` has been deprecated in v1.5+
        # and will be removed in v1.7.
        # Please pass
        # :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with
        # ``process_position``
        # directly to the Trainer's ``callbacks`` argument instead.
        # (type: int, default: 0)
        ##### "process_position": 0,

        # profiling to use, it prints the training profiling
        # at the end of a fit()
        # call
        # default is None, options are simple and advanced,
        # or a custom profiler
        # we don't need it in production
        # To profile individual steps during training and assist in
        # identifying bottlenecks.
        # (type: Union[BaseProfiler, str, null], default: null)
        "profiler": null,

        # refresh rate for the progress bar, default is 1,
        # but 20 is good for Colab
        # ignored if custom callbacks, 0 to disable progress bar
        # we disable it as we are using a custom one
        # How often to refresh progress bar (in steps).
        # Value ``0`` disables progress bar.
        # Ignored when a custom progress bar is passed to
        # :paramref:`~Trainer.callbacks`.
        # Default: None, means
        # a suitable value will be chosen based on the environment
        # (terminal, Google COLAB, etc.).
        # .. deprecated:: v1.5
        #     ``progress_bar_refresh_rate`` has been deprecated in v1.5
        # and will be removed in v1.7.
        #     Please pass
        # :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar`
        # with ``refresh_rate``
        # directly to the Trainer's ``callbacks`` argument instead.
        # To disable the progress bar,
        # pass ``enable_progress_bar = False`` to the Trainer.
        # (type: Optional[int], default: null)
        ##### "progress_bar_refresh_rate": 0,

        # Set to a non-negative integer to reload dataloaders every n epochs.
        # (type: int, default: 0)
        "reload_dataloaders_every_n_epochs": 0,

        # Set to True to reload dataloaders every epoch.
        # .. deprecated:: v1.4
        # ``reload_dataloaders_every_epoch`` has been deprecated in v1.4
        # and will be removed in v1.6.
        # Please use ``reload_dataloaders_every_n_epochs``.
        # (type: bool, default: False)
        ##### "reload_dataloaders_every_epoch": false,

        # add a custom distributed sampler from
        # torch.utils.data.distributed.DistributedSampler for the dataloaders
        # the default uses shuffle=True for training and shuffle=False for
        # test/validation
        # if False, it must be added manually to the dataloaders
        # Explicitly enables or disables sampler replacement. If not specified this
        # will toggled automatically when DDP is used.
        # By default it will add ``shuffle=True`` for
        # train sampler and ``shuffle=False`` for val/test sampler.
        # If you want to customize it,
        # you can set ``replace_sampler_ddp=False`` and add
        # your own distributed sampler.
        # (type: bool, default: True)
        "replace_sampler_ddp": true,

        # to load a checkpoint from which to resume training
        # it always loads the next epoch
        # default is None
        # it can be useful for specific situations
        # Path/URL of the checkpoint from which training is resumed. If there is
        # no checkpoint file at the path, an exception is raised.
        # If resuming from mid-epoch checkpoint,
        # training will start from the beginning of the next epoch.
        # .. deprecated:: v1.5
        # ``resume_from_checkpoint`` is deprecated in v1.5 and will be removed in v1.7.
        # Please pass the path to ``Trainer.fit(..., ckpt_path=...)`` instead.
        # (type: Union[str, Path, null], default: null)
        ##### "resume_from_checkpoint": null,

        # to enable automatic stochastic weight averaging,
        # to improve performance
        # by averaging different runs with different quantizations,
        # check docs for more info
        # default is False
        # we set it to False as there is a corresponding callback with more
        # configurations, hence it will be added there if required
        # Whether to use `Stochastic Weight Averaging (SWA)
        # <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>`_.
        # .. deprecated:: v1.5
        # ``stochastic_weight_avg`` has been deprecated in v1.5 and
        # will be removed in v1.7.
        # Please pass
        # :class:`~pytorch_lightning.callbacks.stochastic_weight_avg.
        # StochasticWeightAveraging`
        # directly to the Trainer's ``callbacks`` argument instead.
        # (type: bool, default: False)
        ##### "stochastic_weight_avg": false,

        # Supports different training strategies with aliases
        # as well custom training type plugins.
        # (type: Union[str, TrainingTypePlugin, null], default: null)
        # here we use gpu
        "strategy": "gpu",

        # to synchronize batch normalization across GPUs
        # Synchronize batch norm layers between process groups/whole world.
        # (type: bool, default: False)
        "sync_batchnorm": false,

        # if True it terminates training with ValueError if any of the loss,
        # accuracy
        # or parameters contain any N(ot)aN(number)
        # default is False, but we want it to True to check
        # if there are any issues
        # If set to True, will terminate training (by raising a `ValueError`) at the
        # end of each training batch, if any of the parameters or the loss are
        # NaN or +/-inf.
        # .. deprecated:: v1.5
        # Trainer argument ``terminate_on_nan`` was deprecated in v1.5
        # and will be removed in 1.7.
        # Please use ``detect_anomaly`` instead.
        # (type: Optional[bool], default: null)
        ##### "terminate_on_nan": true,

        # tracks the norm of the gradients, -1 for no tracking,
        # int for the order
        # default is -1
        # -1 no tracking. Otherwise tracks that p-norm.
        # May be set to 'inf' infinity-norm. If using
        # Automatic Mixed Precision (AMP), the gradients will be unscaled before
        # logging them.
        # (type: Union[int, float, str], default: -1)
        "track_grad_norm": -1,

        # number of TPU cores to use, effective batch size is
        # #TPU cores * batch size
        # int for a single core, list for a set of cores,
        # if bigger than 8 (TPU POD)
        # the script is duplicated and passed to the different core sets
        # default is None, CPU training
        # we don't have TPUs so we set it to None
        # How many TPU cores to train on (1 or 8) / Single TPU to train on [1]
        # (type: Union[int, str, List[int], null], default: null)
        "tpu_cores": null,

        # useful for splitting the backpropagation of a long sequence,
        # useful for
        # time sequences for LSTM
        # default is None, for disabling it
        # NOTE: for more info refer to the official docs and
        # the supporting paper
        "truncated_bptt_steps": null,

        # how often to check the validation dataset
        # if float is a percentage of the training epoch steps,
        # if int it represents
        # the number of batches
        # the use case for the int is when the training sequence is infinite
        # to allow some validation every so often
        # default is 1.0, once per epoch
        # How often to check the validation set.
        # Use float to check within a training epoch,
        # use int to check every n steps (batches).
        # (type: Union[int, float], default: 1.0)
        "val_check_interval": 1.0,

        # the path where to save the weights,
        # overriddenby a checkpoint callback
        # default is os.getcwd()
        # we disable it as we are using a custom checkpoint callback
        # Where to save weights if specified. Will override default_root_dir
        # for checkpoints only. Use this if for whatever reason you need the checkpoints
        # stored in a different place than the logs written in `default_root_dir`.
        # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
        # Defaults to `default_root_dir`.
        # (type: Optional[str], default: null)
        "weights_save_path": null,

        # whether to print a weight summary before training
        # top covers only the high-level modules, full covers
        # all the sub-modules
        # default is top, but we want all the info so we set it to full
        # Prints a summary of the weights when training begins.
        # .. deprecated:: v1.5
        #     ``weights_summary`` has been deprecated in v1.5
        # and will be removed in v1.7.
        #     To disable the summary, pass ``enable_model_summary = False``
        # to the Trainer.
        #     To customize the summary, pass
        # :class:`~pytorch_lightning.callbacks.model_summary.ModelSummary`
        #     directly to the Trainer's ``callbacks`` argument.
        # (type: Optional[str], default: top)
        ##### "weights_summary": null,

    },
}
