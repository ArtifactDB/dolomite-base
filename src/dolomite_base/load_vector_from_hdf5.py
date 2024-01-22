from typing import Sequence, Union
import numpy
import h5py
from biocutils import StringList, IntegerList, FloatList, BooleanList

from . import _utils_string as strings


def load_vector_from_hdf5(handle: h5py.Dataset, curtype: str, convert_to_1darray: bool) -> Union[StringList, IntegerList, FloatList, BooleanList, numpy.ndarray]:
    """
    Load a 
    """
    if curtype == "string":
        values = StringList(strings.load_string_vector_from_hdf5(handle))
        if "missing-value-placeholder" in handle.attrs:
            placeholder = strings.load_scalar_string_attribute_from_hdf5(handle, "missing-value-placeholder")
            for j, y in enumerate(values):
                if y == placeholder:
                    values[j] = None
        return values

    values = handle[:]
    if "missing-value-placeholder" in handle.attrs:
        placeholder = handle.attrs["missing-value-placeholder"]
        if numpy.isnan(placeholder):
            mask = numpy.isnan(values)
        else:
            mask = (values == placeholder)

        if convert_to_1darray:
            return numpy.ma.MaskedArray(_coerce_numpy_type(values, curtype), mask=mask)
        else:
            output = []
            for i, y in enumerate(values):
                if mask[i]:
                    output.append(None)
                else:
                    output.append(y)
            return _choose_NamedList_subclass(output, curtype)

    if convert_to_1darray:
        return _coerce_numpy_type(values, curtype)
    else:
        return _choose_NamedList_subclass(values, curtype)


def _coerce_numpy_type(values: numpy.ndarray, curtype: str) -> numpy.ndarray:
    if curtype == "boolean":
        return values.astype(numpy.bool_)
    elif curtype == "number":
        if not numpy.issubdtype(values.dtype, numpy.floating):
            return values.astype(numpy.double)
    return values


def _choose_NamedList_subclass(values: Sequence, curtype: str) -> Union[IntegerList, FloatList, BooleanList]:
    if curtype == "boolean":
        return BooleanList(values)
    elif curtype == "number":
        return FloatList(values)
    else:
        return IntegerList(values)
