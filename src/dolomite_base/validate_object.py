from . import lib_dolomite_base as lib


def validate_object(path: str):
    """Validate an on-disk representation of an object,
    using validators based on the **takane** specifications.

    Args:
        path: Path to the directory containing the object's representation.

    Raise:
        Error if the validation fails.
    """
    lib.validate(path)
    return
