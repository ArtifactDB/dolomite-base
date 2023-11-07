#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "takane/csv_data_frame.hpp"
#include "takane/hdf5_data_frame.hpp"
#include "byteme/GzipFileReader.hpp"

static std::vector<takane::data_frame::ColumnDetails> configure_columns(
    pybind11::list column_names,
    pybind11::list column_types,
    pybind11::list string_formats,
    pybind11::array_t<bool> factor_ordered,
    pybind11::list factor_levels)
{
    size_t ncols = column_names.size();
    if (ncols != column_types.size()) {
        throw std::runtime_error("'column_names' and 'column_types' should have the same length");
    }
    if (ncols != string_formats.size()) {
        throw std::runtime_error("'column_names' and 'string_formats' should have the same length");
    }
    if (ncols != factor_ordered.size()) {
        throw std::runtime_error("'column_names' and 'factor_ordered' should have the same length");
    }
    if (ncols != factor_levels.size()) {
        throw std::runtime_error("'column_names' and 'factor_levels' should have the same length");
    }

    std::vector<takane::data_frame::ColumnDetails> columns(ncols);
    for (size_t c = 0; c < ncols; ++c) {
        auto& curcol = columns[c];
        curcol.name = column_names[c].cast<std::string>();

        auto curtype = column_types[c].cast<std::string>();
        if (curtype == "integer") {
            curcol.type = takane::data_frame::ColumnType::INTEGER;

        } else if (curtype == "number") {
            curcol.type = takane::data_frame::ColumnType::NUMBER;

        } else if (curtype == "string") {
            curcol.type = takane::data_frame::ColumnType::STRING;
            auto curformat = string_formats[c].cast<std::string>();
            if (curformat == "date") {
                curcol.string_format = takane::data_frame::StringFormat::DATE;
            } else if (curformat == "date-time") {
                curcol.string_format = takane::data_frame::StringFormat::DATE_TIME;
            }

        } else if (curtype == "boolean") {
            curcol.type = takane::data_frame::ColumnType::BOOLEAN;

        } else if (curtype == "factor") {
            curcol.type = takane::data_frame::ColumnType::FACTOR;
            curcol.factor_ordered = factor_ordered.at(c);
            pybind11::list levels(factor_levels[c]);
            auto& col_levels = curcol.factor_levels.mutable_ref();
            for (size_t l = 0, end = levels.size(); l < end; ++l) {
                col_levels.insert(levels[l].cast<std::string>());
            }

        } else if (curtype == "other") {
            curcol.type = takane::data_frame::ColumnType::OTHER;

        } else {
            throw std::runtime_error("as-yet-unsupported type '" + curtype + "'");
        }
    }

    return columns;
}

void check_csv_df(
    std::string path, 
    int nrows,
    bool has_row_names,
    pybind11::list column_names,
    pybind11::list column_types,
    pybind11::list string_formats,
    pybind11::array_t<bool> factor_ordered,
    pybind11::list factor_levels,
    int df_version, // ignored
    bool is_compressed, 
    bool parallel) 
{
    takane::csv_data_frame::Parameters params;
    params.num_rows = nrows;
    params.has_row_names = has_row_names;
    params.columns = configure_columns(column_names, column_types, string_formats, factor_ordered, factor_levels);
    params.parallel = parallel;

    if (is_compressed) {
        byteme::GzipFileReader reader(path);
        takane::csv_data_frame::validate(reader, params);
    } else {
        byteme::RawFileReader reader(path);
        takane::csv_data_frame::validate(reader, params);
    }
}

void check_hdf5_df(
    std::string path, 
    std::string name, 
    int nrows,
    bool has_row_names,
    pybind11::list column_names,
    pybind11::list column_types,
    pybind11::list string_formats,
    pybind11::array_t<bool> factor_ordered,
    pybind11::list factor_levels,
    int df_version, // ignored
    int hdf5_version // ignored
) {
    takane::hdf5_data_frame::Parameters params(std::move(name));
    params.num_rows = nrows;
    params.has_row_names = has_row_names;
    params.columns = configure_columns(column_names, column_types, string_formats, factor_ordered, factor_levels);
    takane::hdf5_data_frame::validate(path.c_str(), params);
}
