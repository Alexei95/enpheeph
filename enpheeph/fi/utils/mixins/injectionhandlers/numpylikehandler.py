import copy
import typing

import src.fi.utils.dataclasses.faultmask
import src.fi.utils.enums.faultmaskop
import src.fi.utils.enums.bitvalue
import src.fi.utils.enums.endianness
import src.fi.utils.mixins.converters.numpylikeconverter


# this class injects faults in numpy-like objects, using the defined converter
class NumpyLikeHandler(
        src.fi.utils.mixins.converters.numpylikeconverter.NumpyLikeConverter
):
    # NOTE: the returned dtype will be the same as the one passed, not the
    # actual dtype for the operation, which must be done manually
    # NOTE: the returned array contains only 1 element
    @classmethod
    def generate_fault_mask(
            cls,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            bit_index: typing.Sequence[int],
            endianness: src.fi.utils.enums.endianness.Endianness,
            bit_value: src.fi.utils.enums.bitvalue.BitValue,
            library: str,
            device: 'cupy.cuda.Device' = None,
    ) -> src.fi.utils.dataclasses.faultmask.FaultMask:
        # if we are using little endian we invert the index, as the LSB is
        # at the end of the list
        # as bitwidth we use the corresponding one for the dtype
        if endianness == endianness.Little:
            bitwidth = cls.numpy_like_bitwidth_from_dtype(
                    dtype=dtype,
                    library=library,
            )
            bit_index = [
                    (bitwidth - 1) - index
                    for index in bit_index
            ]

        # we can set the bits to be set to 1 by using powers of 2
        # which must be then converted to array of the same dtype
        # for StuckAtZero, these bits will be subtracted, as they must be
        # set to 0, otherwise they are set to 1 as default with addition
        # NOTE: dtype must be the same, otherwise we may get casting or
        # errors
        power_of_2_mask = sum(2 ** index for index in bit_index)
        power_of_2_mask_numpy_like = cls.to_numpy_like_array(
                power_of_2_mask,
                dtype=dtype,
                library=library,
        )

        if bit_value in (bit_value.StuckAtOne, bit_value.BitFlip):
            # we generate the base mask, which in this case is all 0s
            # for StuckAtOne and BitFlip we have to set to 1 the chosen bits
            # all the others are 0s
            mask = cls.expand_bit_to_numpy_like_dtype(
                    bit=0,
                    dtype=dtype,
                    library=library,
                    device=device,
            )

            # we add the mask so the selected indices are set to 1
            mask += power_of_2_mask_numpy_like

            if bit_value == bit_value.StuckAtOne:
                mask_op = src.fi.utils.enums.faultmaskop.FaultMaskOp.OR
            elif bit_value == bit_value.BitFlip:
                mask_op = src.fi.utils.enums.faultmaskop.FaultMaskOp.XOR
        elif bit_value == bit_value.StuckAtZero:
            # for StuckAtZero the mask must be all 1s, as we use AND operation
            mask = cls.expand_bit_to_numpy_like_dtype(
                    bit=1,
                    dtype=dtype,
                    library=library,
                    device=device,
            )

            # we subtract the mask as we need the selected indices at 0
            mask -= power_of_2_mask_numpy_like

            mask_op = src.fi.utils.enums.faultmaskop.FaultMaskOp.AND
        else:
            raise ValueError('Unsupported injection type.')
        return src.fi.utils.dataclasses.faultmask.FaultMask(
            mask=mask,
            operation=mask_op,
        )

    # NOTE: the returned dtype will be the same as the one passed, not the
    # actual dtype for the operation, which must be done manually
    @classmethod
    def generate_fault_tensor_mask(
            cls,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            bit_index: typing.Sequence[int],
            tensor_index: typing.Sequence[typing.Union[int, slice]],
            tensor_shape: typing.Sequence[int],
            endianness: src.fi.utils.enums.endianness.Endianness,
            bit_value: src.fi.utils.enums.bitvalue.BitValue,
            library: str,
            device: 'cupy.cuda.Device' = None,
    ) -> src.fi.utils.dataclasses.faultmask.FaultMask:
        element_mask = cls.generate_fault_mask(
                dtype=dtype,
                bit_index=bit_index,
                endianness=endianness,
                bit_value=bit_value,
                library=library,
                device=device,
        )

        # we extend the single-element mask to a whole tensor, covering only
        # the pre-determined indices
        tensor_mask = cls.numpy_like_broadcast(
            element=element_mask.mask,
            index=tensor_index,
            shape=tensor_shape,
            fill_value=element_mask.fill_value,
        )

        return src.fi.utils.dataclasses.faultmask.FaultMask(
            mask=tensor_mask,
            operation=element_mask.operation,
        )

    @classmethod
    def inject_fault_tensor(
            cls,
            numpy_like_element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            bit_index: typing.Sequence[int],
            tensor_index: typing.Sequence[typing.Union[int, slice]],
            tensor_shape: typing.Sequence[int],
            endianness: src.fi.utils.enums.endianness.Endianness,
            bit_value: src.fi.utils.enums.bitvalue.BitValue,
            in_place: bool = True,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        # we gather the information from the element
        dtype = cls.get_numpy_like_dtype(numpy_like_element)
        device = cls.get_numpy_like_device(numpy_like_element)

        # then we create the masks from the bit index
        mask = cls.generate_fault_tensor_mask(
                dtype=dtype,
                bit_index=bit_index,
                tensor_index=tensor_index,
                tensor_shape=tensor_shape,
                endianness=endianness,
                bit_value=bit_value,
                library=cls.get_numpy_like_string(numpy_like_element),
                device=device,
        )

        # we get the injected element, passing in place for creating a copy if
        # needed
        # in_place is handled only here, as all the other things must be
        # recreated anyway
        numpy_like_element_out = cls.inject_fault_tensor_from_mask(
                numpy_like_element=numpy_like_element,
                mask=mask,
                in_place=in_place,
        )

        return numpy_like_element_out

    @classmethod
    def inject_fault_tensor_from_mask(
            cls,
            numpy_like_element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: src.fi.utils.dataclasses.faultmask.FaultMask,
            in_place: bool = True,
    ):
        # if in_place is False, then we generate and return a copy for the
        # element to be injected
        if not in_place:
            numpy_like_element = copy.deepcopy(numpy_like_element)

        # we gather the information about dtypes
        element_dtype = cls.get_numpy_like_dtype(numpy_like_element)
        mask_dtype = cls.get_numpy_like_dtype(mask.mask)

        # we convert the dtypes to uint... to have bitwise operations
        numpy_like_element_bitwise = \
            cls.numpy_like_dtype_to_bitwise_numpy_like(
                    element=numpy_like_element,
            )
        numpy_like_mask_bitwise = cls.numpy_like_dtype_to_bitwise_numpy_like(
                element=mask.mask,
        )

        # we register the associations for the different mask operations
        cls.register(
                mask.operation.AND,
                cls.numpy_like_and_mask_injection,
        )
        cls.register(
                mask.operation.OR,
                cls.numpy_like_or_mask_injection,
        )
        cls.register(
                mask.operation.XOR,
                cls.numpy_like_xor_mask_injection,
        )
        # we dispatch the call for injecting
        # here the operation can be done in-place in all the cases, as we are
        # creating a copy earlier if needed
        injected_numpy_like_element_bitwise = cls.dispatch_call(
                mask.operation,
                element=numpy_like_element_bitwise,
                mask=numpy_like_mask_bitwise,
                in_place=True,
        )
        # we clear the associations
        cls.deregister(
                src.fi.utils.enums.faultmaskop.FaultMaskOp.AND
        )
        cls.deregister(
                src.fi.utils.enums.faultmaskop.FaultMaskOp.OR
        )
        cls.deregister(
                src.fi.utils.enums.faultmaskop.FaultMaskOp.XOR
        )

        # we convert back to the original dtype
        numpy_like_element_out = cls.bitwise_numpy_like_to_numpy_like_dtype(
                element=injected_numpy_like_element_bitwise,
                dtype=element_dtype,
        )
        # we convert back also the element and the mask, to leave them as is
        # since these operations are in-place, we can avoid caring for the
        # returned value
        cls.bitwise_numpy_like_to_numpy_like_dtype(
                    element=numpy_like_element,
                    dtype=element_dtype,
            )
        cls.bitwise_numpy_like_to_numpy_like_dtype(
                    element=numpy_like_mask_bitwise,
                    dtype=mask_dtype,
        )

        return numpy_like_element_out

    # the following classmethods are used to inject faults, and dispatch them
    # properly
    # NOTE: these methods work without knowing the inner workings, only by
    # calling the proper operators, but the type must be converted before-hand
    @classmethod
    def numpy_like_and_mask_injection(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            in_place: bool = True,
    ):
        # if the operation is in-place, we set the output of the operation
        # to be the input, otherwise it is None and it will create a new array
        if in_place:
            element &= mask
            out = element
        else:
            out = element & mask

        return out

    @classmethod
    def numpy_like_or_mask_injection(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            in_place: bool = True,
    ):
        # if the operation is in-place, we set the output of the operation
        # to be the input, otherwise it is None and it will create a new array
        if in_place:
            element |= mask
            out = element
        else:
            out = element | mask

        return out

    @classmethod
    def numpy_like_xor_mask_injection(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            in_place: bool = True,
    ):
        # if the operation is in-place, we set the output of the operation
        # to be the input, otherwise it is None and it will create a new array
        if in_place:
            element ^= mask
            out = element
        else:
            out = element ^ mask

        return out
