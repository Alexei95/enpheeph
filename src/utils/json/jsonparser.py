import collections
import copy
import json
import types
import typing

import src.utils.mixins.dispatcher
import src.utils.mixins.modulegatherer
import src.utils.json.handlers.objecthandler


# to be able to encode JSON, we need to subclass the encoder
# for decoding it is not needed, but we will define a decode classmethod here
class JSONParser(
        src.utils.mixins.modulegatherer.ModuleGatherer,
        json.JSONEncoder,
):
    # dict1 is the base and then dict2 overrides it
    # it similar to the call dict1.update(dict2), but in this case the
    # returned dict is a copy of dict1, so dict1 will stay the same
    # not in-place
    @staticmethod
    def recursive_update_dict(
            dict1: typing.Dict[typing.Any, typing.Any],
            dict2: typing.Dict[typing.Any, typing.Any],
    ) -> typing.Dict[typing.Any, typing.Any]:
        # we copy the first dict to be used as basis
        returned_dict = copy.deepcopy(dict1)
        # we cycle over the elements in the second dict
        for key2, item2 in dict2.items():
            # we get the element with the first key from the first dict
            item1 = returned_dict.get(key2, None)
            # if both the element from dict1 and dict2 are dict we update
            # them in the recursive manner
            if isinstance(item2, collections.abc.Mapping) and isinstance(
                    item1,
                    collections.abc.Mapping
            ):
                updated_item = JSONParser.recursive_update_dict(
                        item1,
                        item2
                )
            # otherwise we just copy it over
            else:
                updated_item = copy.deepcopy(item2)
            # we save the result with a copy of the key
            returned_dict[copy.deepcopy(key2)] = updated_item
        return returned_dict

    DEFAULT_DECODER_STRING = 'default'
    DEFAULT_ENCODER_STRING = 'default'

    DEFAULT_CUSTOM_DECODER_CHECKER_STRING = '__custom__'
    DEFAULT_CUSTOM_DECODER_CHECKER_NEGATED_VALUE = False

    # this classmethod is used to check whether we should use a custom decoder
    # or the default one is suffiecient
    # we check against the default values, so that they can be customized
    # as well as changing the function itself to have a different type of check
    @classmethod
    def DEFAULT_CUSTOM_DECODER_CHECKER_FUNC(
            cls,
            dict_: typing.Dict[str, typing.Any],
            *args,
            **kwargs,
     ) -> bool:
        return dict_.get(
                cls.DEFAULT_CUSTOM_DECODER_CHECKER_STRING,
                cls.DEFAULT_CUSTOM_DECODER_CHECKER_NEGATED_VALUE,
        ) is not cls.DEFAULT_CUSTOM_DECODER_CHECKER_NEGATED_VALUE

    DEFAULT_CUSTOM_DECODER_KEY = '__custom_decoder__'

    # this classmethod is used to get the function to be used when decoding
    # from the dict
    # as for the checker, it can be customized or substituted entirely
    @classmethod
    def DEFAULT_CUSTOM_DECODER_GETTER(
            cls,
            dict_: typing.Dict[str, typing.Any],
            *args,
            **kwargs,
     ) -> bool:
        decoder_name = dict_.get(
                cls.DEFAULT_CUSTOM_DECODER_KEY,
                cls.DEFAULT_DECODER_STRING,
        )
        return decoder_name

    # we use two instances of Dispatcher, one for the decoders and one for the
    # encoders
    DecoderDispatcher = src.utils.mixins.dispatcher.Dispatcher()
    EncoderDispatcher = src.utils.mixins.dispatcher.Dispatcher()

    def __init__(
            self,
            default_decoder: typing.Callable[
                    [typing.Dict[typing.Any, typing.Any]],
                    typing.Any
            ] = src.utils.json.handlers.objecthandler.ObjectHandler.from_json,
            default_encoder: typing.Callable[
                    [typing.Any],
                    typing.Dict[str, typing.Any]
            ] = src.utils.json.handlers.objecthandler.ObjectHandler.to_json,
    ):
        super().__init__()
        self.DecoderDispatcher.register(
                self.DEFAULT_DECODER_STRING,
                default_decoder
        )
        self.EncoderDispatcher.register(
                self.DEFAULT_ENCODER_STRING,
                default_encoder
        )

    # the method has to be a normal instance method as it uses the default
    # value set in the __init__
    # this method loads different streams in JSON formats, with the first
    # being the lowest priority and the last being the highest one
    def load_streams(
            self,
            fps: typing.Sequence[typing.IO],
            # extra arguments are passed to json.load
            *args,
            **kwargs,
    ) -> typing.Dict[str, typing.Any]:
        returned_dict = {}
        for fp in fps:
            d = json.load(fp, *args, object_hook=self.decode, **kwargs)
            # we update the current dict with the newly-parsed JSON
            returned_dict = self.recursive_update_dict(returned_dict, d)
        return returned_dict

    # the method has to be a normal instance method as it uses the default
    # value set in the __init__
    # this method loads the different strings interpreting them as JSON
    # the first one is the least priority
    def load_strings(
            self,
            strings: typing.Sequence[str],
            # extra arguments are passed to json.load
            *args,
            **kwargs,
    ) -> typing.Dict[str, typing.Any]:
        returned_dict = {}
        for string in strings:
            d = json.loads(string, *args, object_hook=self.decode, **kwargs)
            if isinstance(d, collections.abc.Mapping):
                # we update the current dict with the newly-parsed JSON
                returned_dict = self.recursive_update_dict(returned_dict, d)
            else:
                # if not we simply overwrite the value
                returned_dict = d
        return returned_dict

    # the method has to be a normal instance method as it uses the default
    # value set in the __init__
    def decode(self, d):
        if self.DEFAULT_CUSTOM_DECODER_CHECKER_FUNC(d):
            decoder_name = self.DEFAULT_CUSTOM_DECODER_GETTER(d)
            return self.DecoderDispatcher.dispatch_call(
                    decoder_name,
                    d
            )
        else:
            return d
