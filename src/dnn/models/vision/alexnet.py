import torch
import torchvision


class AlexNet(torchvision.models.AlexNet):
    def init_weights(self):
        # this initialization is similar to the ResNet one
        # taken from https://github.com/Lornatang/AlexNet-PyTorch/
        # @ alexnet_pytorch/model.py#L63
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                        m.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
