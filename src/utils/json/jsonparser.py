import collections
import copy
import json
import pathlib
import types
import typing

import src.utils.instance_or_classmethod
import src.utils.mixins.dispatcher
import src.utils.mixins.modulegatherer


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

    DEFAULT_CUSTOM_FUNCTION_CHECKER_STRING = '__custom__'
    DEFAULT_CUSTOM_FUNCTION_CHECKER_NEGATED_VALUE = False

    # this classmethod is used to check whether we should use a custom decoder
    # or the default one is suffiecient
    # we check against the default values, so that they can be customized
    # as well as changing the function itself to have a different type of check
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def DEFAULT_CUSTOM_FUNCTION_CHECKER(
            self_or_cls,
            dict_: typing.Dict[str, typing.Any],
            *args,
            **kwargs,
     ) -> bool:
        try:
            flag = dict_.get(
                    self_or_cls.DEFAULT_CUSTOM_FUNCTION_CHECKER_STRING,
                    self_or_cls.DEFAULT_CUSTOM_FUNCTION_CHECKER_NEGATED_VALUE,
            ) is not self_or_cls.DEFAULT_CUSTOM_FUNCTION_CHECKER_NEGATED_VALUE
        # if we trigger the exception it means the dict_ is not actually a dict
        # so we return False
        except AttributeError:
            flag = False

        return flag

    DEFAULT_CUSTOM_DECODER_KEY = '__custom_decoder__'
    DEFAULT_CUSTOM_DECODER_NULL_NAME = None

    # this classmethod is used to get the function to be used when decoding
    # from the dict
    # as for the checker, it can be customized or substituted entirely
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def DEFAULT_CUSTOM_DECODER_GETTER(
            self_or_cls,
            dict_: typing.Dict[str, typing.Any],
            *args,
            **kwargs,
     ) -> bool:
        # if the dict_ parameter is not a dict we still return the default null
        # name
        try:
            decoder_name = dict_.get(
                    self_or_cls.DEFAULT_CUSTOM_DECODER_KEY,
                    self_or_cls.DEFAULT_CUSTOM_DECODER_NULL_NAME,
            )
        except AttributeError:
            decoder_name = self_or_cls.DEFAULT_CUSTOM_DECODER_NULL_NAME
        return decoder_name

    DEFAULT_CUSTOM_POSTPROCESSOR_KEY = '__custom_postprocessor__'
    DEFAULT_CUSTOM_POSTPROCESSOR_NULL_NAME = None

    # this classmethod is used to get the function to be used when
    # postprocessing from the dict
    # as for the checker, it can be customized or substituted entirely
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def DEFAULT_CUSTOM_POSTPROCESSOR_GETTER(
            self_or_cls,
            dict_: typing.Union[typing.Dict[str, typing.Any], typing.Any],
            *args,
            **kwargs,
     ) -> bool:
        # if the dict has no get attribute, then we return the default
        try:
            postprocessor_name = dict_.get(
                    self_or_cls.DEFAULT_CUSTOM_POSTPROCESSOR_KEY,
                    self_or_cls.DEFAULT_CUSTOM_POSTPROCESSOR_NULL_NAME,
            )
        except AttributeError:
            return self_or_cls.DEFAULT_CUSTOM_POSTPROCESSOR_NULL_NAME
        return postprocessor_name

    # we use two instances of Dispatcher, one for the decoders and one for the
    # encoders
    DecoderDispatcher = src.utils.mixins.dispatcher.Dispatcher()
    PostprocessorDispatcher = src.utils.mixins.dispatcher.Dispatcher()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.import_submodules(
                package_name='src.utils.json.handlers',
                package_path=pathlib.Path(
                        __file__
                ).resolve().parent / 'handlers',
                root=pathlib.Path(
                        __file__
                ).resolve().parent.parent.parent.parent,
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
        # post-processing
        return self.post_process_decoding(
                returned_dict,
                complete=returned_dict
        )

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
        # post-processing
        return self.post_process_decoding(
                returned_dict,
                complete=returned_dict
        )

    def load_paths(
            self,
            paths: typing.Sequence[pathlib.Path],
            # extra arguments are passed to json.load
            *args,
            **kwargs,
    ) -> typing.Dict[str, typing.Any]:
        # we read the paths to get the strings to feed load_strings
        strings = [p.read_text() for p in paths]
        return self.load_strings(strings, *args, **kwargs)

    # the method has to be a normal instance method as it uses the default
    # value set in the __init__
    # we make it into a instance- or class- method so that it can be run on
    # a class- or instance- basis
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def decode(self_or_cls, d):
        # if there is a definition for a custom function
        if self_or_cls.DEFAULT_CUSTOM_FUNCTION_CHECKER(d):
            # we get the name of the decoder
            decoder_name = self_or_cls.DEFAULT_CUSTOM_DECODER_GETTER(d)
            # if the name is None or the default we skip it
            if decoder_name is self_or_cls.DEFAULT_CUSTOM_DECODER_NULL_NAME:
                return d
            return self_or_cls.DecoderDispatcher.dispatch_call(
                    decoder_name,
                    d
            )
        else:
            return d

    @src.utils.instance_or_classmethod.instance_or_classmethod
    def postprocess(self_or_cls, d, complete):
        # if there is a definition for a custom function
        if self_or_cls.DEFAULT_CUSTOM_FUNCTION_CHECKER(d):
            # we get the name of the postprocessor
            postprocessor_name = self_or_cls.\
                    DEFAULT_CUSTOM_POSTPROCESSOR_GETTER(d)
            # if the name is None or the default we skip it
            if postprocessor_name is self_or_cls.\
                    DEFAULT_CUSTOM_POSTPROCESSOR_NULL_NAME:
                return d
            return self_or_cls.PostprocessorDispatcher.dispatch_call(
                    postprocessor_name,
                    d,
                    complete
            )
        else:
            return d

    @src.utils.instance_or_classmethod.instance_or_classmethod
    def post_process_decoding(
            self_or_cls,
            element: typing.Any,
            complete: typing.Any,
    ):
        # if the element is a dict, we iteratively go through it
        if isinstance(element, collections.abc.Mapping):
            updated_element = {}
            # we cycle through all the elements in the dict
            for key, item in element.items():
                # we first go recursively inside
                updated_item = self_or_cls.post_process_decoding(
                        item,
                        complete
                )
                updated_item = self_or_cls.postprocess(updated_item, complete)
                updated_element[key] = updated_item
        else:
            updated_element = element

        return self_or_cls.postprocess(updated_element, complete)
