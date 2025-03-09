from typing import Optional, Tuple, List
import h5py
import numpy
import biocutils


def save_fixed_length_strings(handle: h5py.Group, name: str, x: List[str]) -> h5py.Dataset:
    """Save a list of strings into a fixed-length string dataset.

    Args:
        handle:     
            Handle to a HDF5 Group.
        
        name: 
            Name of the dataset to create in ``handle``.
        
        x: 
            List of strings to save.

    Returns:
        ``x`` is saved into the group as a fixed-length string dataset,
        and a NumPy dataset handle is returned.
    """
    tmp = [ y.encode("UTF-8") for y in x ]
    maxed = 1
    for b in tmp:
        if len(b) > maxed:
            maxed = len(b)
    return handle.create_dataset(name, data=tmp, dtype="S" + str(maxed), compression="gzip", chunks=True)


def load_string_vector_from_hdf5(handle: h5py.Dataset) -> List[str]:
    output = handle[:]

    if len(output):
        _output = []
        for x in output:
            _out = x.decode("UTF-8") if isinstance(x, (bytes, numpy.bytes_)) else x
            _output.append(_out)
        output = _output
    return output


def load_scalar_string_attribute_from_hdf5(handle, name: str) -> str:
    output = handle.attrs[name]
    if isinstance(output, (bytes, numpy.bytes_)):
        output = output.decode("UTF-8")
    return output


def collect_stats(x_encoded: list) -> Tuple:
    maxed = 1
    total = 0
    for b in x_encoded:
        bn = len(b)
        total += bn
        if bn > maxed:
            maxed = bn
    return maxed, total


def use_vls(maxed: int, total: int, nstr: int) -> bool:
    return (maxed * nstr > total + nstr * 16)


def dump_vls(ghandle: h5py.Group, pointers: str, heap: str, x_encoded: list, placeholder: Optional[str]):
    dtype = numpy.dtype([('offset', 'u8'), ('length', 'u8')])

    nstr = len(x_encoded)
    x_pointers = [None] * nstr
    cumulative = 0
    for i, b in enumerate(x_encoded):
        bn = len(b)
        x_pointers[i] = (cumulative, bn)
        cumulative += bn

    phandle = ghandle.create_dataset(pointers, data=x_pointers, dtype=dtype, compression="gzip", chunks=True)
    if placeholder is not None:
        phandle.attrs["missing-value-placeholder"] = placeholder

    x_heap = numpy.ndarray(cumulative, dtype=numpy.dtype("u1"))
    cumulative = 0
    for i, b in enumerate(x_encoded):
        start = cumulative
        cumulative += len(b)
        x_heap[start:cumulative] = list(b)
    ghandle.create_dataset(heap, data=x_heap, dtype='u1', compression="gzip", chunks=True)


def read_vls(ghandle: h5py.Group, pointers: str, heap: str, as_numpy: bool):
    pset = ghandle[pointers]
    placeholder = None 
    if "missing-value-placeholder" in pset.attrs:
        placeholder = load_scalar_string_attribute_from_hdf5(pset, "missing-value-placeholder")

    heap = ghandle[heap]
    all_pointers = pset[:]
    all_heap = heap[:]
    output = [None] * len(all_pointers)
    for i, payload in enumerate(all_pointers):
        start, length = payload
        output[i] = bytes(all_heap[start:start + length]).decode("UTF-8")

    if as_numpy:
        output = numpy.array(output)
        if placeholder is not None:
            mask = output == placeholder
            output = numpy.ma.MaskedArray(output, mask=mask)
    else:
        if placeholder is not None:
            for j, y in enumerate(output):
                if y == placeholder:
                    output[j] = None
        output = biocutils.StringList(output)

    return output
