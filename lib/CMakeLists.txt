cmake_minimum_required(VERSION 3.24)

project(dolomite_base
    VERSION 1.0.0
    DESCRIPTION "Building the dolomite-base shared library"
    LANGUAGES CXX)

# Defining the targets.
set(TARGET dolomite_base)

find_package(pybind11 CONFIG REQUIRED)

# pybind11 method:
pybind11_add_module(${TARGET} 
    src/load_list.cpp
    src/validate.cpp
    src/init.cpp
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

# Need to set this to get access to filesystem.
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)

target_include_directories(${TARGET} PRIVATE "${ASSORTHEAD_INCLUDE_DIR}")
target_link_libraries(${TARGET} PRIVATE pybind11::pybind11)

#set(ZLIB_USE_STATIC_LIBS ON)
#find_package(ZLIB REQUIRED CONFIG CMAKE_FIND_ROOT_PATH_BOTH)

find_package(HDF5 COMPONENTS C CXX REQUIRED)
target_link_libraries(${TARGET} PRIVATE hdf5::hdf5 hdf5::hdf5_cpp)
target_link_libraries(${TARGET} PRIVATE libaec::aec-static libaec::sz-static) # Static HDF5 doesn't include its transitive dependencies.

# Need to use find_library() as our new Zlib doesn't come with cmake install scripts.
find_library(ZLIB_PATH libz.a ${CMAKE_PREFIX_PATH})
target_link_libraries(${TARGET} PRIVATE ${ZLIB_PATH})

set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME lib_dolomite_base
    PREFIX ""
)

if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE H5_BUILT_AS_DYNAMIC_LIB=1)
endif()
