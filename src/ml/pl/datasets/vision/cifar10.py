import pl_bolts


class CIFAR10DataModule(pl_bolts.datamodules.CIFAR10DataModule):
    def __init__(
            self,
            train_transforms=None,
            test_transforms=None,
            val_transforms=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if train_transforms is not None:
            self.train_transforms = train_transforms
        if test_transforms is not None:
            self.test_transforms = test_transforms
        if val_transforms is not None:
            self.val_transforms = val_transforms
