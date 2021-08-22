import pathlib

import norse
import torch


class NCARSConvModel(torch.nn.Module):
    def __init__(
            self,
            trainable_parameters=False,
            debug=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._debug = debug

        if trainable_parameters:
            lif1 = norse.torch.LIFCell(
                    p=norse.torch.LIFParameters(
                            tau_syn_inv=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 30, 25],
                                            fill_value=(
                                                    norse.torch.LIFParameters.
                                                    _field_defaults.get(
                                                            "tau_syn_inv"
                                                    )
                                            ),
                                    ),
                            ),
                            tau_mem_inv=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 30, 25],
                                            fill_value=(
                                                    norse.torch.LIFParameters.
                                                    _field_defaults.get(
                                                            "tau_mem_inv"
                                                    )
                                            ),
                                    ),
                            ),
                            v_leak=torch.nn.Parameter(
                                    norse.torch.LIFParameters.
                                    _field_defaults.get(
                                            "v_leak"
                                    )
                            ),
                            v_th=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 30, 25],
                                            fill_value=(
                                                    0.4
                                                    # norse.torch.LIFParameters.
                                                    # _field_defaults.get(
                                                    #         "v_th"
                                                    # )
                                            ),
                                    ),
                            ),
                            v_reset=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 30, 25],
                                            fill_value=(
                                                    norse.torch.LIFParameters.
                                                    _field_defaults.get(
                                                            "v_reset"
                                                    )
                                            ),
                                    ),
                            ),
                            alpha=torch.tensor(
                                    norse.torch.LIFParameters.
                                    _field_defaults.get(
                                            "alpha"
                                    )
                            ),
                            method="super",
                    ),
                    dt=0.01,
            )
            lif2 = norse.torch.LIFCell(
                    p=norse.torch.LIFParameters(
                            tau_syn_inv=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 15, 12],
                                            fill_value=(
                                                    norse.torch.LIFParameters.
                                                    _field_defaults.get(
                                                            "tau_syn_inv"
                                                    )
                                            ),
                                    ),
                            ),
                            tau_mem_inv=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 15, 12],
                                            fill_value=(
                                                    norse.torch.LIFParameters.
                                                    _field_defaults.get(
                                                            "tau_mem_inv"
                                                    )
                                            ),
                                    ),
                            ),
                            v_leak=torch.nn.Parameter(
                                    norse.torch.LIFParameters.
                                    _field_defaults.get(
                                            "v_leak"
                                    )
                            ),
                            v_th=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 15, 12],
                                            fill_value=(
                                                    0.4
                                                    # norse.torch.LIFParameters.
                                                    # _field_defaults.get(
                                                    #         "v_th"
                                                    # )
                                            ),
                                    ),
                            ),
                            v_reset=torch.nn.Parameter(
                                    torch.full(
                                            size=[32, 15, 12],
                                            fill_value=(
                                                    norse.torch.LIFParameters.
                                                    _field_defaults.get(
                                                            "v_reset"
                                                    )
                                            ),
                                    ),
                            ),
                            alpha=torch.tensor(
                                    norse.torch.LIFParameters.
                                    _field_defaults.get(
                                            "alpha"
                                    )
                            ),
                            method="super",
                    ),
                    dt=0.01,
            )
            li = norse.torch.LICell(
                    p=norse.torch.LIParameters(
                            tau_syn_inv=torch.nn.Parameter(
                                    torch.full(
                                            size=[2],
                                            fill_value=(
                                                    norse.torch.LIParameters.
                                                    _field_defaults.get(
                                                            "tau_syn_inv"
                                                    )
                                            ),
                                    ),
                            ),
                            tau_mem_inv=torch.nn.Parameter(
                                    torch.full(
                                            size=[2],
                                            fill_value=(
                                                    norse.torch.LIParameters.
                                                    _field_defaults.get(
                                                            "tau_mem_inv"
                                                    )
                                            ),
                                    ),
                            ),
                            v_leak=torch.nn.Parameter(
                                    norse.torch.LIParameters.
                                    _field_defaults.get(
                                            "v_leak"
                                    )
                            ),
                    ),
                    dt=torch.nn.Parameter(
                            torch.full(
                                    size=[2],
                                    fill_value=0.01,
                            ),
                    ),
            )
        else:
            lif1 = norse.torch.LIFCell()
            lif2 = norse.torch.LIFCell()
            li = norse.torch.LICell()

        self.sequential = norse.torch.module.SequentialState(

                torch.nn.AvgPool2d(
                        kernel_size=4,
                        stride=4,
                        padding=0,
                        ceil_mode=False,
                ),
                torch.nn.Dropout(
                        p=0.1,
                        inplace=False,
                ),
                torch.nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        dilation=1,
                        stride=1,
                        groups=1,
                ),

                lif1,
                torch.nn.AvgPool2d(
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        ceil_mode=False,
                ),
                torch.nn.Dropout(
                        p=0.1,
                        inplace=False,
                ),
                torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        dilation=1,
                        stride=1,
                        groups=1,
                ),
                lif2,

                torch.nn.AvgPool2d(
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        ceil_mode=False,
                ),
                torch.nn.Dropout(
                        p=0.2,
                        inplace=False,
                ),

                torch.nn.Flatten(
                        start_dim=1,
                        end_dim=-1,
                ),

                torch.nn.Linear(
                        in_features=1344,
                        out_features=500,
                        bias=True,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                        in_features=500,
                        out_features=2,
                        bias=True,
                ),
                li,

        )

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
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, state=None):
        if self._debug:
            with pathlib.Path('snn_ncars_conv_debug.txt').open('a') as f:
                f.write("###########\n")
                f.write(f'LIF tau syn inv  {torch.mean(self.sequential[3].p.tau_syn_inv):.15f} {torch.std(self.sequential[3].p.tau_syn_inv):.15f}\n')
                f.write(f'LIF tau mem inv  {torch.mean(self.sequential[3].p.tau_mem_inv):.15f} {torch.std(self.sequential[3].p.tau_mem_inv):.15f}\n')
                f.write(f'LIF v leak       {torch.mean(self.sequential[3].p.v_leak):.15f} {torch.std(self.sequential[3].p.v_leak):.15f}\n')
                f.write(f'LIF v th         {torch.mean(self.sequential[3].p.v_th):.15f} {torch.std(self.sequential[3].p.v_th):.15f}\n')
                f.write(f'LIF v reset      {torch.mean(self.sequential[3].p.v_reset):.15f} {torch.std(self.sequential[3].p.v_reset):.15f}\n')
                f.write("###\n")
                f.write(f'LIF2 tau syn inv {torch.mean(self.sequential[7].p.tau_syn_inv):.15f} {torch.std(self.sequential[7].p.tau_syn_inv):.15f}\n')
                f.write(f'LIF2 tau mem inv {torch.mean(self.sequential[7].p.tau_mem_inv):.15f} {torch.std(self.sequential[7].p.tau_mem_inv):.15f}\n')
                f.write(f'LIF2 v leak      {torch.mean(self.sequential[7].p.v_leak):.15f} {torch.std(self.sequential[7].p.v_leak):.15f}\n')
                f.write(f'LIF2 v th        {torch.mean(self.sequential[7].p.v_th):.15f} {torch.std(self.sequential[7].p.v_th):.15f}\n')
                f.write(f'LIF2 v reset     {torch.mean(self.sequential[7].p.v_reset):.15f} {torch.std(self.sequential[7].p.v_reset):.15f}\n')
                f.write("###\n")
                f.write(f'LI tau syn inv   {torch.mean(self.sequential[-1].p.tau_syn_inv):.15f} {torch.std(self.sequential[-1].p.tau_syn_inv):.15f}\n')
                f.write(f'LI tau mem inv   {torch.mean(self.sequential[-1].p.tau_mem_inv):.15f} {torch.std(self.sequential[-1].p.tau_mem_inv):.15f}\n')
                f.write(f'LI v leak        {torch.mean(self.sequential[-1].p.v_leak):.15f} {torch.std(self.sequential[-1].p.v_leak):.15f}\n')
                f.write(f'LI dt            {torch.mean(self.sequential[-1].dt):.15f} {torch.std(self.sequential[-1].dt):.15f}\n')
                f.write("###########\n")
                f.flush()

        return self.sequential.forward(x, state=state)
