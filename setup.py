"""ByCycle setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('bycycle', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = [req for req in requirements_file.read().splitlines()]

setup(
    name = 'bycycle',
    version = __version__,
    description = 'Cycle-by-cycle analyses of neural oscillations.',
    long_description = long_description,
    python_requires = '>=3.5',
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    url = 'https://github.com/bycycle-tools/bycycle',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    platforms = 'any',
    project_urls = {
        'Documentation' : 'https://bycycle-tools.github.io/bycycle/',
        'Bug Reports' : 'https://github.com/bycycle-tools/bycycle/issues',
        'Source' : 'https://github.com/bycycle-tools/bycycle'
    },
    download_url = 'https://github.com/bycycle-tools/bycycle/releases',
    keywords = ['neuroscience', 'neural oscillations', 'waveform', 'shape', 'electrophysiology'],
    install_requires = install_requires,
    tests_require = ['pytest']
)
