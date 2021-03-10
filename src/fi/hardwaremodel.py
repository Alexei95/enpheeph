class HardwareModel(object):
    # number of kernels we can run concurrently
    kernel_slots: int = 1024
    # in this dict we have the
    kernel_scheduling: typing.Dict[float, Kernel]
