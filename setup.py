"""bycycle setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('bycycle', 'version.py')) as vf:
    exec(vf.read())

# Load long descritption from README.rst
with open('README.rst') as f:
    long_description = f.read()

setup(
    name = 'bycycle',
    version = __version__,
    description = 'cycle-by-cycle analysis of neural oscillations',
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
        'Operating System :: MacOS :: MacOS X',
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
    install_requires = ['numpy', 'scipy', 'pandas'],
    tests_require = ['pytest'],
    extras_require = {
        'plot'    : ['matplotlib'],
        'tests'   : ['pytest'],
        'all'     : ['matplotlib', 'pytest']
    }
)
