#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy>=1.16.3",
    "scipy>=1.2.0",
    "ephem>=3.7.6.0",
    "healpy>=1.14.0",
    "scikit-sparse>=0.4.5",
    "pint-pulsar>=0.8.2",
    "libstempo>=2.4.0",
    "enterprise-pulsar>=3.1.0",
    "emcee",
    "ptmcmcsampler",
]

test_requirements = []

# Extract version


def get_version():
    with open("enterprise_extensions/__init__.py") as f:
        for line in f.readlines():
            if "__version__" in line:
                return line.split('"')[1]


setup(
    name="enterprise_extensions",
    version=get_version(),
    description="Extensions, model shortcuts, and utilities for the enterprise PTA analysis framework.",
    long_description=readme + "\n\n" + history,
    long_description_content_type='text/x-rst',
    classifiers=[
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="gravitational-wave, black-hole binary, pulsar-timing arrays",
    url="https://github.com/stevertaylor/enterprise_extensions",
    author="Stephen R. Taylor, Paul T. Baker, Jeffrey S. Hazboun, Sarah Vigeland",
    author_email="srtaylor@caltech.edu",
    license="MIT",
    packages=[
        "enterprise_extensions",
        "enterprise_extensions.frequentist",
        "enterprise_extensions.chromatic",
    ],
    package_data={
        "enterprise_extensions.chromatic": [
            "ACE_SWEPAM_daily_proton_density_1998_2018_MJD_cm-3.txt"
        ]
    },
    test_suite="tests",
    tests_require=test_requirements,
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
