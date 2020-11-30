import typing

from . import common
from ...sources import common as particle_common


class ChipModel(object):
    # NOTE: for now we support only 2 dimensions
    chip_map: typing.List[typing.List[common.SiliconType]]
    # NOTE: this is the dimension of a chip element as defined in chip_map
    #       for now it is based around 1cm^2, but it will be variable in the
    #       future
    chip_cell_dimension: float
    # threshold for having a fault corresponding to a paricle hit
    # NOTE: in future, we could have a distribution for energy of particle hit,
    #       inside the model of the silicon-particle interaction, and then
    #       there could be different thresholds for having a fault depending on
    #       the type of transistor that is being hit, e.g. register, memory, ..
    # BUG: for now we use the energy to compute the area of impact, and it has
    #      the same energy throughout the impact area
    particle_energy_to_fault_threshold: float

    def __init__(
            self, chip_map, chip_cell_dimension,
            particle_energy_to_fault_threshold, *args, **kwargs):
        kwargs.update({
            'chip_map': chip_map,
            'chip_cell_dimension': chip_cell_dimension,
            'particle_energy_to_fault_threshold':
            particle_energy_to_fault_threshold})

        super().__init__(*args, **kwargs)

        self.chip_map = chip_map
        self.chip_cell_dimension = chip_cell_dimension
        self.particle_energy_to_fault_threshold = particle_energy_to_fault_threshold

    def particle_to_fault_conversion(self, particle):




    # NOTE: here we convert the input particles into a list of real faults
    # each fault should contain the type of fault
    def __call__(self, *args, **kwargs):
        pass
