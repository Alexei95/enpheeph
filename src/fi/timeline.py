from . import summary

Operation =

class Timeline(object):
    # we need summary to have MACs, execution time, layers and operations
    model_summary: summary.Summary
    # we have a dict mapping each time range to a list of operations
    _timeline: typing.OrderedDict[]
