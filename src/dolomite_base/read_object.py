from typing import Any, Optional
from importlib import import_module
import os
import json


read_object_registry = {
    "atomic_vector": "dolomite_base.read_atomic_vector",
    "string_factor": "dolomite_base.read_string_factor",
    "simple_list": "dolomite_base.read_simple_list",
    "data_frame": "dolomite_base.read_data_frame",
    "dense_array": "dolomite_matrix.read_dense_array",
    "compressed_sparse_matrix": "dolomite_matrix.read_compressed_sparse_matrix",
    "genomic_ranges": "dolomite_ranges.read_genomic_ranges",
    "genomic_ranges_list": "dolomite_ranges.read_genomic_ranges_list",
    "sequence_information": "dolomite_ranges.read_sequence_information",
    "summarized_experiment": "dolomite_se.read_summarized_experiment",
    "range_summarized_experiment": "dolomite_se.read_ranged_summarized_experiment",
    "single_cell_experiment": "dolomite_sce.read_single_cell_experiment",
    "multi_sample_dataset": "dolomite_mae.read_multi_assay_experiment",
}


def read_object(path: str, metadata: Optional[dict] = None, **kwargs) -> Any:
    """
    Read an object from its on-disk representation. This will dispatch to
    individual reading functions - possibly from different packages in the
    **dolomite** framework based on the ``metadata`` from the ``OBJECT`` file. 

    Application developers can control the dispatch process by setting the
    ``read_object_registry`` for the relevant object type(s). The value can
    either be a string specifying the fully qualified name of a function
    (including all modules) or a function that accepts the same arguments as
    ``read_object`` (no need to consider ``metadata = None``).

    Args:
        path: 
            Path to a directory containing the object.

        metadata: 
            Metadata for the object. This is read from the `OBJECT` file if None.

        kwargs: 
            Further arguments, passed to individual methods.

    Returns:
        Some kind of object.
    """
    if metadata is None:
        with open(os.path.join(path, "OBJECT"), "rb") as handle:
            metadata = json.load(handle)

    tt = metadata["type"]
    if tt not in read_object_registry:
        raise NotImplementedError("could not find a Python function to read '" + tt + "'")

    command = read_object_registry[tt]
    if isinstance(command, str): 
        first_period = command.find(".")
        mod_name = command[:first_period]

        try:
            mod = import_module(mod_name)
        except:
            raise ModuleNotFoundError("no module named '" + mod_name + "' for reading an instance of '" + tt + "'")

        command = getattr(mod, command[first_period + 1:])
        read_object_registry[tt] = command

    return command(path, metadata, **kwargs)
