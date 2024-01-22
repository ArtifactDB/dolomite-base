import numpy
from biocutils import Factor
import h5py

from . import _utils_string as strings


def save_factor_to_hdf5(handle: h5py.Group, f: Factor):
    strings.save_fixed_length_strings(handle, "levels", f.get_levels())

    codes = f.get_codes()
    is_missing = codes == -1
    has_missing = is_missing.any()
    nlevels = len(f.get_levels())
    if has_missing:
        codes = codes.astype(numpy.uint32, copy=True)
        codes[is_missing] = nlevels

    dhandle = handle.create_dataset("codes", data=codes, dtype="u4", compression="gzip", chunks=True)
    if has_missing:
        dhandle.attrs.create("missing-value-placeholder", data=nlevels, dtype="u4")

    if f.get_ordered():
        handle.attrs.create("ordered", data=1, dtype="i1")


def load_factor_from_hdf5(handle: h5py.Group):
    chandle = handle["codes"]
    codes = chandle[:]
    codes = codes.astype(numpy.int32, copy=False)

    if "missing-value-placeholder" in chandle.attrs:
        placeholder = chandle.attrs["missing-value-placeholder"]
        codes[codes == placeholder] = -1

    ordered = False
    if "ordered" in handle.attrs:
        ordered = handle.attrs["ordered"][()] != 0
    
    levels = strings.load_string_vector_from_hdf5(handle["levels"])
    return Factor(codes, levels, ordered = ordered)
