# Changelog

## Version 0.4.5

- Added preliminary support for custom variable length string (VLS) arrays, which compress the heap for more efficient storage than HDF5's VLS implementation.
This is enabled in the `save_object()` methods for `StringList`s, `BiocFrame`s or lists via the `*_string_list_vls=` option.

## Version 0.4.4

- More fixes to the wheel building.
- Removed unnecessary dependency on **jsonschema**.

## Version 0.4.3

- Bugfix for the `read_object()` dispatcher on `ranged_summarized_experiment`.
- Implemented `validate_directory()` to validate all objects in a directory.
- Implemented `list_objects()` to list all objects in a directory.
- Removed unnecessary dependency on **dolomite-schemas**.

## Version 0.4.2

- More fixes to the wheel building.

## Version 0.4.1

- Fixes to the wheel building.

## Version 0.4.0

- chore: Remove Python 3.8 (EOL).
- precommit: Replace docformatter with ruff's formatter.

## Version 0.3.0

- Fixes to support NumPy 2.0 release.

## Version 0.0.1 - 0.2.4

- Use the Python restoration commands from the schemas.
- First release onto PyPI
