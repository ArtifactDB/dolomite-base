"""
    Setup file for dolomite_base.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.5.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
from glob import glob
import pathlib
import os
import shutil
import sys


def define_install_dir():
    if "DOLOMITE_INSTALLED_PATH" in os.environ:
        return os.environ["DOLOMITE_INSTALLED_PATH"]
    else:
        return os.path.join(os.getcwd(), "installed")


def define_custom_builder(builder):
    if builder is not None:
        return builder
    import subprocess
    class Tmp:
        def __init__(self):
            pass
        def spawn(self, cmd):
            subprocess.run(cmd, check=True)
    return Tmp()


def build_aec(builder):
    install_dir = define_install_dir()
    if os.path.exists(install_dir) and os.path.exists(os.path.join(install_dir, ".AEC")):
        return install_dir

    builder = define_custom_builder(builder)

    version = "v1.1.4"
    if not os.path.exists("extern"):
        os.mkdir("extern")

    src_dir = os.path.join("extern", "libaec-" + version)
    if not os.path.exists(src_dir):
        tarball = os.path.join("extern", "libaec.tar.gz")
        if not os.path.exists(tarball):
            import urllib.request
            target_url = "https://gitlab.dkrz.de/k202009/libaec/-/archive/" + version + "/libaec-" + version + ".tar.gz" 
            urllib.request.urlretrieve(target_url, tarball)
        import tarfile
        with tarfile.open(tarball, "r") as tf:
            tf.extractall("extern")

    build_dir = os.path.join("extern", "build-libaec-" + version)
    if not os.path.exists(install_dir):
        os.mkdir("installed")

    cmd = [
        "cmake",
        "-S", src_dir,
        "-B", build_dir,
        "-DCMAKE_POSITION_INDEPENDENT_CODE=true",
        "-DCMAKE_INSTALL_PREFIX=" + install_dir
    ]
    if os.name != "nt":
        cmd.append("-DCMAKE_BUILD_TYPE=Release")
    if "MORE_CMAKE_OPTIONS" in os.environ:
        cmd += os.environ["MORE_CMAKE_OPTIONS"].split()
    builder.spawn(cmd)

    cmd = ['cmake', '--build', build_dir]
    if os.name == "nt":
        cmd += ["--config", "Release"]
    builder.spawn(cmd)
    cmd = ['cmake', '--install', build_dir]
    builder.spawn(cmd)

    with open(os.path.join(install_dir, ".AEC"), "w") as handle:
        handle.write("")
    return install_dir


def build_hdf5(builder):
    install_dir = define_install_dir()
    if os.path.exists(install_dir) and os.path.exists(os.path.join(install_dir, ".HDF5")):
        return install_dir

    builder = define_custom_builder(builder)

    version = "2.0.0"
    if not os.path.exists("extern"):
        os.mkdir("extern")

    src_dir = os.path.join("extern", "hdf5-" + version)
    if not os.path.exists(src_dir):
        tarball = os.path.join("extern", "hdf5.tar.gz")
        if not os.path.exists(tarball):
            import urllib.request
            target_url = "https://github.com/HDFGroup/hdf5/releases/download/" + version + "/hdf5-" + version + ".tar.gz"
            urllib.request.urlretrieve(target_url, tarball)
        import tarfile
        with tarfile.open(tarball, "r") as tf:
            tf.extractall("extern")

    build_dir = os.path.join("extern", "build-hdf5-" + version)
    if not os.path.exists(install_dir):
        os.mkdir("installed")

    cmd = [
        "cmake",
        "-S", src_dir,
        "-B", build_dir,
        "-DCMAKE_POSITION_INDEPENDENT_CODE=true",
        "-DCMAKE_INSTALL_PREFIX=" + install_dir,
        "BUILD_TESTING=OFF",
        "BUILD_SHARED_LIBS=OFF",
        "HDF5_BUILD_CPP_LIB=ON",
        "HDF5_BUILD_TOOLS=OFF",
        "HDF5_BUILD_EXAMPLES=OFF",
        "HDF5_ENABLE_ZLIB_SUPPORT=ON",
        "HDF5_ENABLE_SZIP_SUPPORT=ON",
        "HDF5_USE_LIBAEC_STATIC=ON",
        "CMAKE_PREFIX_PATH=" + install_dir
    ]
    if os.name != "nt":
        cmd.append("-DCMAKE_BUILD_TYPE=Release")
    if "MORE_CMAKE_OPTIONS" in os.environ:
        cmd += os.environ["MORE_CMAKE_OPTIONS"].split()
    builder.spawn(cmd)

    cmd = ['cmake', '--build', build_dir]
    if os.name == "nt":
        cmd += ["--config", "Release"]
    builder.spawn(cmd)
    cmd = ['cmake', '--install', build_dir]
    builder.spawn(cmd)

    with open(os.path.join(install_dir, ".HDF5"), "w") as handle:
        handle.write("")
    return install_dir


## Adapted from https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py.
class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        build_aec(self)
        install_dir = build_hdf5(self)

        build_temp = pathlib.Path(self.build_temp)
        build_lib = pathlib.Path(self.build_lib)
        outpath = os.path.join(build_lib.absolute(), ext.name)

        if not os.path.exists(build_temp):
            import assorthead
            import pybind11
            cmd = [ 
                "cmake", 
                "-S", "lib",
                "-B", str(build_temp),
                "-Dpybind11_DIR=" + os.path.join(os.path.dirname(pybind11.__file__), "share", "cmake", "pybind11"),
                "-DPYTHON_EXECUTABLE=" + sys.executable,
                "-DASSORTHEAD_INCLUDE_DIR=" + assorthead.includes(),
                "-DCMAKE_PREFIX_PATH=" + install_dir
            ]
            if os.name != "nt":
                cmd.append("-DCMAKE_BUILD_TYPE=Release")
                cmd.append("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + outpath)

            if "MORE_CMAKE_OPTIONS" in os.environ:
                cmd += os.environ["MORE_CMAKE_OPTIONS"].split()
            self.spawn(cmd)

        if not self.dry_run:
            cmd = ['cmake', '--build', str(build_temp)]
            if os.name == "nt":
                cmd += ["--config", "Release"]
            self.spawn(cmd)
            if os.name == "nt": 
                # Gave up trying to get MSVC to respect the output directory.
                # Delvewheel also needs it to have a 'pyd' suffix... whatever.
                shutil.copyfile(os.path.join(build_temp, "Release", "_core.dll"), os.path.join(outpath, "_core.pyd"))

if __name__ == "__main__":
    import os
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[CMakeExtension("dolomite_base")],
            cmdclass={
                'build_ext': build_ext
            }
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
