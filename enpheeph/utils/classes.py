import ast
import typing

InstanceGeneratorBaseClass = typing.TypeVar(
        "InstanceGeneratorBaseClass",
        bound="InstanceGeneratorMixin"
)


class InstanceGeneratorMixin(object):
    # this method does not work with enums or with nested function calls
    @classmethod
    def from_safe_repr(
            cls: typing.Type[InstanceGeneratorBaseClass],
            representation: str
    ) -> InstanceGeneratorBaseClass:
        # we assume only one call
        call_element = ast.parse(representation).body[0].value
        # we assume function  name is identical to the cls.__qualname__
        assert cls.__qualname__ == call_element.func.id
        # we parse arguments and keyword-arguments using ast.literal_eval
        args = []
        for arg in call_element.args:
            args.append(ast.literal_eval(arg))
        kwargs = {}
        for keyword_element in call_element.keywords:
            kwargs[keyword_element.arg] = ast.literal_eval(
                    keyword_element.value,
            )