from setuptools import setup, find_packages, Extension
from os.path import join
from DectrisTools import VERSION
import numpy

setup(
    name="DectrisTools",
    version=VERSION,
    packages=find_packages(),
    ext_modules=[
        Extension(
            name="DectrisTools.lib.computation",
            sources=[join("DectrisTools", "lib", "computation.c")],
            include_dirs=[numpy.get_include()],
        )
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "h5py",
        "hdf5plugin",
        "pyqtgraph",
        "PyQt5",
        "pillow",
        "tqdm",
        "numba",
        "uedinst@git+https://github.com/Siwick-Research-Group/uedinst.git",
    ],
    url="https://github.com/kremeyer/DectrisTools",
    license="GNU General Public License v3.0",
    author="Laurenz Kremeyer",
    author_email="laurenz.kremeyer@mail.mcgill.ca",
    description="tools for the Dectris Quadro detector",
)
