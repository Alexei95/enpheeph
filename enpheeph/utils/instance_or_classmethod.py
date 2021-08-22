class instance_or_classmethod(classmethod):
    def __get__(self, instance, type_):
        # if there is no instance we give the operation to the classmethod
        # otherwise we use the current instance function getter
        descr_get = (
                super().__get__
                if instance is None
                else self.__func__.__get__
        )
        return descr_get(instance, type_)
