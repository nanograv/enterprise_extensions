from setuptools import setup

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
    long_description="See: `github.com/stevertaylor/enterprise_extensions <https://github.com/stevertaylor/enterprise_extensions>`_." ,
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
    author='Stephen R. Taylor, Paul T. Baker, Jeffrey S. Hazboun',
    author_email='srtaylor@caltech.edu',
    license='MIT',
    packages=['enterprise_extensions'],
    install_requires=['numpy','scipy','enterprise'],
    include_package_data=True,
    zip_safe=False,
)
