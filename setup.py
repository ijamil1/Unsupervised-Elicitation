# setup.py
from setuptools import find_packages, setup

setup(
    name="UE",  # Replace with your project name
    version="0.1",
    packages=find_packages(where="."),  # Find all packages in the repo
    package_dir={
        "": ".",  # Root of the repo is the package source
    },
    install_requires=[
        # List your dependencies here, e.g.,
        # "numpy",
        # "torch",
    ],
    include_package_data=True,  # Include non-Python files if needed
)
