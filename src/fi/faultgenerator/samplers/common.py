import collections

DEFAULT_SAMPLER_INDEX_CLASS = collections.namedtuple('SamplerIndex',
                                                     ['tensor_index',
                                                      'bit_index'])
# we can overwrite docstrings, for main class docs we need to append
DEFAULT_SAMPLER_INDEX_CLASS.__doc__ += 'Default class for returning a bit to flip from a Sampler.'
DEFAULT_SAMPLER_INDEX_CLASS.tensor_index.__doc__ = 'Index used to access the tensor to target for the flipping.'
DEFAULT_SAMPLER_INDEX_CLASS.bit_index.__doc__ = 'Index used to access the targeted bit inside a tensor.'
