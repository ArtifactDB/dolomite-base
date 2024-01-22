from typing import Sequence, Union
import numpy
import h5py
from biocutils import StringList, IntegerList, FloatList, BooleanList

from . import _utils_string as strings


def load_vector_from_hdf5(handle: h5py.Dataset, expected_type: type, report_1darray: bool) -> Union[StringList, IntegerList, FloatList, BooleanList, numpy.ndarray]:
    """
    Load a vector from a 1-dimensional HDF5 dataset, with coercion to the expected type.
    Any missing value placeholders are used to set Nones or to create masks.

    Args:
        handle: Handle to a HDF5 dataset.

        expected_type: 
            Expected type of the output vector. This should be one of
            ``float``, ``int``, ``str`` or ``bool``.

        report_1darray:
            Whether to report the output as a 1-dimensional NumPy array.

    Returns:
        The contents of the dataset as a vector-like object. By default, this
        is a typed :py:class:`~biocutils.biocutils.NamedList` subclass with
        missing values represented by None. If ``keep_as_1darray = True``, a
        1-dimensional NumPy array is returned instead, possibly with masking.
    """
    if expected_type == str:
        values = strings.load_string_vector_from_hdf5(handle)
        placeholder = None
        if "missing-value-placeholder" in handle.attrs:
            placeholder = strings.load_scalar_string_attribute_from_hdf5(handle, "missing-value-placeholder")
        if report_1darray:
            values = numpy.array(values)
            if placeholder is not None:
                mask = values == placeholder
                values = numpy.ma.MaskedArray(values, mask=mask)
        else:
            if placeholder is not None:
                for j, y in enumerate(values):
                    if y == placeholder:
                        values[j] = None
            values = StringList(values)
        return values

    values = handle[:]
    if "missing-value-placeholder" in handle.attrs:
        placeholder = handle.attrs["missing-value-placeholder"]
        if numpy.isnan(placeholder):
            mask = numpy.isnan(values)
        else:
            mask = (values == placeholder)

        if report_1darray:
            return numpy.ma.MaskedArray(_coerce_numpy_type(values, expected_type), mask=mask)
        else:
            output = []
            for i, y in enumerate(values):
                if mask[i]:
                    output.append(None)
                else:
                    output.append(y)
            return _choose_NamedList_subclass(output, expected_type)

    if report_1darray:
        return _coerce_numpy_type(values, expected_type)
    else:
        return _choose_NamedList_subclass(values, expected_type)


def _coerce_numpy_type(values: numpy.ndarray, expected_type: str) -> numpy.ndarray:
    if expected_type == bool:
        return values != 0
    elif expected_type == float:
        if not numpy.issubdtype(values.dtype, numpy.floating):
            return values.astype(numpy.double)
    return values


def _choose_NamedList_subclass(values: Sequence, expected_type: str) -> Union[IntegerList, FloatList, BooleanList]:
    if expected_type == bool:
        return BooleanList(values)
    elif expected_type == float:
        return FloatList(values)
    else:
        return IntegerList(values)
