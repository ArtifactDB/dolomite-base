from biocframe import BiocFrame
import h5py
import os

from .alt_read_object import alt_read_object
from . import _utils_string as strings
from .load_vector_from_hdf5 import load_vector_from_hdf5
from ._utils_factor import load_factor_from_hdf5 
from . import _utils_misc as misc


def read_data_frame(path: str, metadata: dict, data_frame_represent_numeric_column_as_1darray : bool = True, **kwargs) -> BiocFrame:
    """Load a data frame from a HDF5 file. In general, this function should not
    be called directly but instead via :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: 
            Path to the directory containing the object.

        metadata: 
            Metadata for the object.

        data_frame_represent_numeric_column_as_1darray: 
            Whether numeric columns should be represented as 1-dimensional
            NumPy arrays. This is more efficient than regular Python lists but
            discards the distinction between vectors and 1-D arrays. Usually
            this is not an important difference, but nonetheless, users can set
            this flag to ``False`` to load columns as (typed) lists instead.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A data frame.
    """
    column_names = []
    contents = {}
    row_names = None
    expected_rows = 0

    with h5py.File(os.path.join(path, "basic_columns.h5"), "r") as handle:
        ghandle = handle["data_frame"]
        expected_rows = ghandle.attrs["row-count"][()]
        column_names = strings.load_string_vector_from_hdf5(ghandle["column_names"])
        if "row_names" in ghandle:
            row_names = strings.load_string_vector_from_hdf5(ghandle["row_names"])

        dhandle = ghandle["data"]
        for i, col in enumerate(column_names):
            name = str(i)
            if name not in dhandle:
                contents[col] = alt_read_object(os.path.join(path, "other_columns", name), **kwargs)
            else:
                xhandle = dhandle[name]
                curtype = strings.load_scalar_string_attribute_from_hdf5(xhandle, "type")
                if curtype == "factor":
                    contents[col] = load_factor_from_hdf5(xhandle)
                else:
                    expected_type = misc.translate_type(curtype)
                    contents[col] = load_vector_from_hdf5(xhandle, expected_type, report_1darray=(expected_type != str and data_frame_represent_numeric_column_as_1darray))

    df = BiocFrame(contents, number_of_rows=expected_rows, row_names=row_names, column_names=column_names)

    other_dir = os.path.join(path, "other_annotations")
    if os.path.exists(other_dir):
        df.set_metadata(alt_read_object(other_dir, **kwargs).as_dict(), in_place=True)

    mcol_dir = os.path.join(path, "column_annotations")
    if os.path.exists(mcol_dir):
        df.set_column_data(alt_read_object(mcol_dir, **kwargs), in_place=True)

    return df
