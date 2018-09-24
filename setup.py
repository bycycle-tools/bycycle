"""bycycle setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('bycycle', 'version.py')) as vf:
    exec(vf.read())

# Copy in long description.
#  Note: this is a partial copy from the README
#    Only update here in coordination with the README, to keep things consistent.
long_description = \
"""
=======================================================
bycycle: cycle-by-cycle analysis of neural oscillations
=======================================================

bycycle is a python implementation of a cycle-by-cycle approach to analyzing
neural oscillations (see Cole & Voytek, 2018, biorxiv). This approach
quantifies features of neural oscillations in the time domain as opposed
to the frequency domain. Rather than applying narrowband filters and other
methods that utilize a sinusoidal basis, this characterization segments a
recording into individual cycles and directly measures each of their properties
including amplitude, period, and symmetry. This is most advantageous for
analyzing the waveform shape properties of neural oscillations, but it may also
provide advantages for studying traditional amplitude and frequency effects, as
well. It also implements burst detection, which has been gaining traction
recently (see e.g. Jones, 2016, COiN) so that we only analyze oscillatory
properties when there is indeed an oscillation.

If you use this code in your project, please cite:

Cole SR & Voytek B (2018) Cycle-by-cycle analysis of neural oscillations. bioRxiv, 302000.
doi: https://doi.org/10.1101/302000

Paper Link: https://www.biorxiv.org/content/early/2018/04/16/302000
"""

setup(
    name = 'bycycle',
    version = __version__,
    description = 'cycle-by-cycle analysis of neural oscillations',
    long_description = long_description,
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    url = 'https://github.com/voytekresearch/bycycle',
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
    ],
    download_url = 'https://github.com/voytekresearch/bycycle/releases',
    keywords = ['neuroscience', 'neural oscillations', 'waveform', 'shape', 'electrophysiology'],
    install_requires = ['numpy', 'scipy', 'pandas'],
    tests_require = ['pytest'],
    extras_require = {
        'plot'    : ['matplotlib'],
        'tests'   : ['pytest'],
        'all'     : ['matplotlib', 'pytest']
    }
)