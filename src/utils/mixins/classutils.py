import typing


class ClassUtils(object):
    # this method returns the main library (the first module) associated to an
    # object, by going through its class, the module and then returning the
    # first module name
    @classmethod
    def get_main_library_from_object(cls, object: typing.Any):
        return object.__class__.__module__.split('.')[0]
