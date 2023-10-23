#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "takane/csv_data_frame.hpp"
#include "takane/hdf5_data_frame.hpp"
#include "byteme/GzipFileReader.hpp"

static std::vector<takane::data_frame::ColumnDetails> configure_columns(
    pybind11::list column_names,
    pybind11::array_t<int32_t> column_types,
    pybind11::array_t<int32_t> string_formats,
    pybind11::list factor_levels)
{
    size_t ncols = column_names.size();
    if (ncols != column_types.size()) {
        throw std::runtime_error("'column_names' and 'column_types' should have the same length");
    }
    if (ncols != string_formats.size()) {
        throw std::runtime_error("'column_names' and 'string_formats' should have the same length");
    }
    if (ncols != factor_levels.size()) {
        throw std::runtime_error("'column_names' and 'factor_levels' should have the same length");
    }

    std::vector<takane::data_frame::ColumnDetails> columns(ncols);
    for (size_t c = 0; c < ncols; ++c) {
        auto& curcol = columns[c];
        curcol.name = column_names[c].cast<std::string>();

        auto curtype = column_types.at(c);
        if (curtype == 0) {
            curcol.type = takane::data_frame::ColumnType::INTEGER;

        } else if (curtype == 1) {
            curcol.type = takane::data_frame::ColumnType::NUMBER;

        } else if (curtype == 2) {
            curcol.type = takane::data_frame::ColumnType::STRING;
            auto curformat = string_formats.at(c);
            if (curformat == 1) {
                curcol.format = takane::data_frame::StringFormat::DATE;
            } else if (curformat == 2) {
                curcol.format = takane::data_frame::StringFormat::DATE_TIME;
            }

        } else if (curtype == 3) {
            curcol.type = takane::data_frame::ColumnType::BOOLEAN;

        } else if (curtype == 4) {
            curcol.type = takane::data_frame::ColumnType::FACTOR;
            pybind11::list levels(factor_levels[c]);
            for (size_t l = 0, end = levels.size(); l < end; ++l) {
                curcol.add_factor_level(levels[l].cast<std::string>());
            }

        } else if (curtype == 5) {
            curcol.type = takane::data_frame::ColumnType::OTHER;

        } else {
            throw std::runtime_error("unknown type code '" + std::to_string(curtype) + "'");
        }
    }

    return columns;
}

void check_csv_df(
    std::string path, 
    int nrows,
    bool has_row_names,
    pybind11::list column_names,
    pybind11::array_t<int32_t> column_types,
    pybind11::array_t<int32_t> string_formats,
    pybind11::list factor_levels,
    int df_version,
    bool is_compressed, 
    bool parallel) 
{
    comservatory::ReadOptions opt;
    opt.parallel = parallel;
    auto columns = configure_columns(column_names, column_types, string_formats, factor_levels);

    if (is_compressed) {
        byteme::GzipFileReader reader(path);
        takane::data_frame::validate_csv(reader, nrows, has_row_names, columns, opt, df_version);
    } else {
        byteme::RawFileReader reader(path);
        takane::data_frame::validate_csv(reader, nrows, has_row_names, columns, opt, df_version);
    }
}

void check_hdf5_df(
    std::string path, 
    std::string name, 
    int nrows,
    bool has_row_names,
    pybind11::list column_names,
    pybind11::array_t<int32_t> column_types,
    pybind11::array_t<int32_t> string_formats,
    pybind11::list factor_levels,
    int df_version,
    int hdf5_version)
{
    auto columns = configure_columns(column_names, column_types, string_formats, factor_levels);
    takane::data_frame::validate_hdf5(path, name, nrows, has_row_names, columns, df_version, hdf5_version);
}
