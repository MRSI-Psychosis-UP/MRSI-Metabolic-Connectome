import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mrsitoolbox",
    version = "3.0.0",
    description = ("Analysis toolbox for MRSI data"),
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=find_packages(include=['tools','graphplot','filters',
                                    'registration','connectomics',
                                    'bids']),
    # long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Topic :: DataUtils",
        "License :: OSI Approved :: BSD License",
    ],
)
