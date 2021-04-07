#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy',
                'scipy',]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

ext_modules=[
    Extension('jitterext',
             ['./enterprise_extensions/outlier/jitterext.pyx'],
             include_dirs = [numpy.get_include()],
             extra_compile_args=["-O2"]),
    Extension('choleskyext_omp',
             ['./enterprise_extensions/outlier/choleskyext_omp.pyx'],
             include_dirs = [numpy.get_include()],
             extra_link_args=["-liomp5"],
             extra_compile_args=["-O2", "-fopenmp", "-fno-wrapv"])
]

# Extract version
def get_version():
    with open('enterprise_extensions/models.py') as f:
        for line in f.readlines():
            if "__version__" in line:
                return line.split('"')[1]

setup(
    name='enterprise_extensions',
    version=get_version(),
    description='Extensions, model shortcuts, and utilities for the enterprise PTA analysis framework.',
    long_description=readme + '\n\n' + history,
    classifiers=[
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='gravitational-wave, black-hole binary, pulsar-timing arrays',
    url='https://github.com/stevertaylor/enterprise_extensions',
    author='Stephen R. Taylor, Paul T. Baker, Jeffrey S. Hazboun, Sarah Vigeland',
    author_email='srtaylor@caltech.edu',
    license='MIT',
    packages=['enterprise_extensions',
              'enterprise_extensions.frequentist',
              'enterprise_extensions.chromatic',
              'enterprise_extensions.outlier'],
    package_data={'enterprise_extensions.chromatic':
                  ['ACE_SWEPAM_daily_proton_density_1998_2018_MJD_cm-3.txt']},
    ext_modules = cythonize(ext_modules),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
