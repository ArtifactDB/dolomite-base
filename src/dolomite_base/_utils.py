from numpy import ndarray


def _fragment_string_contents(strlengths: ndarray, buffer: ndarray) -> list[str]:
    sofar = 0
    collected = []
    for i, x in enumerate(strlengths):
        endpoint = sofar + x 
        collected.append(buffer[sofar:endpoint].decode("ASCII"))
        sofar = endpoint
    return collected


def _mask_strings(collected: list, mask: ndarray):
    for i, x in enumerate(mask):
        if x:
            collected[i] = None

