# -*- coding: utf-8 -*-
import re


CAMEL_TO_SNAKE_REGEX: re.Pattern[str] = re.compile(
    "((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))"
)


# this function is required to convert CamelCase to snake_case
def camel_to_snake(camel: str) -> str:
    # from https://stackoverflow.com/a/12867228
    return CAMEL_TO_SNAKE_REGEX.sub(r"_\1", camel).lower()
