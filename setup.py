from setuptools import setup, find_packages
from DectrisTools import VERSION

setup(
    name="DectrisTools",
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pyqtgraph",
        "PyQt5",
        "pillow",
        "tqdm",
        "uedinst@git+https://github.com/Siwick-Research-Group/uedinst.git",
    ],
    url="https://github.com/kremeyer/DectrisTools",
    license="",
    author="Laurenz Kremeyer",
    author_email="laurenz.kremeyer@mail.mcgill.ca",
    description="tools for the Dectris Quadro detector",
)
