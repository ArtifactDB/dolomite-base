cmake_minimum_required(VERSION 3.24)

project(dolomite_base
    VERSION 1.0.0
    DESCRIPTION "Building the dolomite-base shared library"
    LANGUAGES CXX)

# Importing all of the dependencies with pinned versions (even for transitive dependencies).
include(FetchContent)

FetchContent_Declare(
  millijson
  GIT_REPOSITORY https://github.com/ArtifactDB/millijson
  GIT_TAG v1.0.1
)

FetchContent_Declare(
  byteme 
  GIT_REPOSITORY https://github.com/LTLA/byteme
  GIT_TAG v1.1.0
)

FetchContent_Declare(
  uzuki2
  GIT_REPOSITORY https://github.com/ArtifactDB/uzuki2
  GIT_TAG v1.4.0
)

FetchContent_Declare(
  ritsuko 
  GIT_REPOSITORY https://github.com/ArtifactDB/ritsuko
  GIT_TAG v0.5.2
)

FetchContent_Declare(
  chihaya
  GIT_REPOSITORY https://github.com/ArtifactDB/chihaya
  GIT_TAG v1.1.0
)

FetchContent_Declare(
  takane
  GIT_REPOSITORY https://github.com/ArtifactDB/takane
  GIT_TAG v0.6.1
)

FetchContent_MakeAvailable(uzuki2)
FetchContent_MakeAvailable(byteme)
FetchContent_MakeAvailable(millijson)
FetchContent_MakeAvailable(takane)
FetchContent_MakeAvailable(ritsuko)
FetchContent_MakeAvailable(chihaya)

# Defining the targets.
set(TARGET dolomite_base)

find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(${TARGET} 
    src/load_list.cpp
    src/validate.cpp
    src/init.cpp
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

target_link_libraries(${TARGET} PRIVATE uzuki2 takane pybind11::pybind11)

set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME lib_dolomite_base
    PREFIX ""
)

if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE H5_BUILT_AS_DYNAMIC_LIB=1)
endif()
