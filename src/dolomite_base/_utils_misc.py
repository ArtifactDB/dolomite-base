LIMIT32 = 2**31


def scalar_exceeds_int32(x: int) -> bool:
    return x < -LIMIT32 or x >= LIMIT32


def sequence_exceeds_int32(x: int, check_none: bool = True) -> bool:
    if check_none:
        for y in x:
            if y is not None and scalar_exceeds_int32(y):
                return True
    else:
        for y in x:
            if scalar_exceeds_int32(y):
                return True
    return False


def translate_type(t: str) -> type:
    if t == "string":
        return str
    elif t == "number":
        return float 
    elif t == "integer":
        return int
    elif t == "boolean":
        return bool
    else:
        raise NotImplementedError("unknown vector type '" + t + "'")
