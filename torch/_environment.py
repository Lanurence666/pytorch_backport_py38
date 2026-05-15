from typing_extensions import Literal


def is_fbcode() -> Literal[False]:
    return False


def is_prod() -> Literal[False]:
    return False
