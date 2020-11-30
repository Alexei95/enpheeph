from ... import common as fault_common


class ArchitecturalFault(fault_common.FaultABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
