from setuptools import setup, find_packages, Extension
from os.path import join
from DectrisTools import VERSION

setup(
    name="DectrisTools",
    version=VERSION,
    packages=find_packages(),
    ext_modules=[
        Extension(
            name="DectrisTools.lib.computation",
            sources=[join("DectrisTools", "lib", "computation.c")],
        )
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "pyqtgraph",
        "PyQt5",
        "pillow",
        "tqdm",
        "numba",
        "psutil",
        "uedinst@git+https://github.com/Siwick-Research-Group/uedinst.git",
    ],
    url="https://github.com/kremeyer/DectrisTools",
    license="",
    author="Laurenz Kremeyer",
    author_email="laurenz.kremeyer@mail.mcgill.ca",
    description="tools for the Dectris Quadro detector",
)
