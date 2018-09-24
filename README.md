# bycycle - cycle-by-cycle analysis of neural oscillations

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Latest Version](https://img.shields.io/pypi/v/bycycle.svg)](https://pypi.python.org/pypi/bycycle/)
[![Build Status](https://travis-ci.org/voytekresearch/bycycle.svg)](https://travis-ci.org/voytekresearch/bycycle)
[![License](https://img.shields.io/pypi/l/bycycle.svg)](https://opensource.org/licenses/Apache-2.0)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bycycle.svg)](https://pypi.python.org/pypi/bycycle/)

## Overview

bycycle is a python implementation of a cycle-by-cycle approach to analyzing neural oscillations ([Cole & Voytek, 2018](https://www.biorxiv.org/content/early/2018/04/16/302000)). This approach quantifies features of neural oscillations in the time domain as opposed to the frequency domain. Rather than applying narrowband filters and other methods that utilize a sinusoidal basis, this characterization segments a recording into individual cycles and directly measures each of their properties including amplitude, period, and symmetry. This is most advantageous for analyzing the waveform shape properties of neural oscillations, but it may also provide advantages for studying traditional amplitude and frequency effects, as well. It also implements burst detection, which has been gaining traction recently (see e.g. [Jones, 2016](https://www.sciencedirect.com/science/article/pii/S0959438816300769?via%3Dihub)) so that we only analyze oscillatory properties when there is indeed an oscillation.

A full description of the method and approach is available in the paper below.

## Reference

If you use this code in your project, please cite [this preprint](https://www.biorxiv.org/content/early/2018/04/16/302000):

    Cole SR & Voytek B (2018) Cycle-by-cycle analysis of neural oscillations. bioRxiv, 302000.
    doi: https://doi.org/10.1101/302000

## Dependencies

bycycle is written in Python, and is tested to work with Python 3.5

It has the following dependencies:
- numpy
- scipy
- pandas
- matplotlib (optional)
- pytest (optional)

## Matlab Support

Coming soon.

## Install

To install the latest stable release of bycycle, you can use pip:

`$ pip install bycycle`

## Development Branch

To get the lastest, development version, you can get the code using git:

`$ git clone https://github.com/voytekresearch/bycycle`

To then install the development version (without making changes to it), move into the directory you cloned and run:

`$ pip install .`

Otherwise, if you want to install an editable, development version, move into the directory you cloned and install with:

$ pip install -e .

## Usage

Coming soon

## Output

Coming soon
