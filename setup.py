"""
    Setup file for dolomite_base.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.5.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup, Extension
import assorthead
from glob import glob


if __name__ == "__main__":
    import os
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[
                Extension(
                    "dolomite_base._core",
                    sorted(glob("src/dolomite_base/lib/*.cpp")),
                    include_dirs=[
                        assorthead.includes(),
                        "src/dolomite_base/include"
                    ],
                    language="c++",
                    extra_compile_args=[
                        "-std=c++17",
                    ],
                    extra_link_args=[
                        "-lz"
                    ],
                )
            ]
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
