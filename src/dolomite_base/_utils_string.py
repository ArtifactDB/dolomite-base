from typing import List
import h5py
import numpy


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
