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

## Adapted from https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py.
class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()
        os.chdir(ext.name)
        if not os.path.exists("build"):
            self.spawn([
                'cmake',
                "-S", ".",
                "-B", "build",
                "-DCMAKE_BUILD_TYPE=Release"
            ])

        if not self.dry_run:
            self.spawn(['cmake', '--build', 'build'])
            for x in os.listdir("build"):
                if x.startswith('_core'):
                    shutil.copyfile(os.path.join("build", x), os.path.join(cwd, "src/dolomite_base", x))
                    break

        os.chdir(str(cwd))


if __name__ == "__main__":
    import os
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[CMakeExtension("lib")],
            cmdclass={'build_ext': build_ext}
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
