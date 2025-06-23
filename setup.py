#!/usr/bin/env python

import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    author="Federico Lucchetti",
    author_email='federico.lucchetti@unil.ch',
    python_requires='>=3.6',
    name = "mrsitoolbox",
    version = "1.0.0",
    description = ("Analysis toolbox for MRSI data"),
    license = "License :: Other/Proprietary License",
    include_package_data=True,
    keywords='mrsitoolbox',
    url='https://github.com/MRSI-Psychosis-UP/Metabolic-Connectome.git',
    packages=find_packages(include=['tools','graphplot','filters',
                                    'registration','connectomics',
                                    'randomize']),
    # long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "License :: Other/Proprietary License"
    ],
)



