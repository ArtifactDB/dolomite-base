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
  GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(millijson)

FetchContent_Declare(
  byteme 
  GIT_REPOSITORY https://github.com/LTLA/byteme
  GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(byteme)

FetchContent_Declare(
  uzuki2
  GIT_REPOSITORY https://github.com/ArtifactDB/uzuki2
  GIT_TAG v1.1.3
)
FetchContent_MakeAvailable(uzuki2)

FetchContent_Declare(
  comservatory
  GIT_REPOSITORY https://github.com/ArtifactDB/comservatory
  GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(comservatory)

# Defining the targets.
set(TARGET dolomite_base)

add_library(${TARGET} SHARED
    src/bindings.cpp
    src/load_csv.cpp
    src/load_list.cpp
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

target_link_libraries(${TARGET} uzuki2 comservatory)

set_target_properties(dolomite_base PROPERTIES 
    OUTPUT_NAME _core 
    PREFIX ""
)

if (HDF5_USE_STATIC_LIBRARIES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(SZIP NAMES sz szip)
    target_link_libraries(${TARGET} ${SZIP})
endif()