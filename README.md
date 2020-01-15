# bycycle - cycle-by-cycle analysis of neural oscillations

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Latest Version](https://img.shields.io/pypi/v/bycycle.svg)](https://pypi.python.org/pypi/bycycle/)
[![Build Status](https://travis-ci.org/bycycle-tools/bycycle.svg)](https://travis-ci.org/bycycle-tools/bycycle)
[![License](https://img.shields.io/pypi/l/bycycle.svg)](https://opensource.org/licenses/Apache-2.0)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bycycle.svg)](https://pypi.python.org/pypi/bycycle/)
[![codecov](https://codecov.io/gh/bycycle-tools/bycycle/branch/master/graph/badge.svg)](https://codecov.io/gh/bycycle-tools/bycycle)

## Overview

bycycle is a python implementation of a cycle-by-cycle approach to analyzing neural oscillations ([Cole & Voytek, J Neurophysiol 2019](https://www.physiology.org/doi/abs/10.1152/jn.00273.2019)). This approach quantifies features of neural oscillations in the time domain as opposed to the frequency domain. Rather than applying narrowband filters and other methods that utilize a sinusoidal basis, this characterization segments a recording into individual cycles and directly measures each of their properties including amplitude, period, and symmetry. This is most advantageous for analyzing the waveform shape properties of neural oscillations, but it may also provide advantages for studying traditional amplitude and frequency effects, as well. It also implements burst detection, which has been gaining traction recently (see e.g. [Jones, 2016](https://www.sciencedirect.com/science/article/pii/S0959438816300769?via%3Dihub)) so that we only analyze oscillatory properties when there is indeed an oscillation.

A full description of the method and approach is available in the paper below.

## Reference

If you use this code in your project, please cite [our paper](https://www.physiology.org/doi/abs/10.1152/jn.00273.2019):

    Cole SR & Voytek B (2019) Cycle-by-cycle analysis of neural oscillations. J Neurophysiol 122:2, 849-861.
    doi: https://doi.org/10.1152/jn.00273.2019
	
The preprint of the paper is also available at: https://www.biorxiv.org/content/early/2018/04/16/302000

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

To get the lastest, development version, you can get the code using git:

`$ git clone https://github.com/bycycle-tools/bycycle`

To then install the development version (without making changes to it), move into the directory you cloned and run:

`$ pip install .`

Otherwise, if you want to install an editable, development version, move into the directory you cloned and install with:

$ pip install -e .

## Usage

The main function in `bycycle` is `compute_features()`, which takes a time series and some parameters as inputs and returns a table of features for each cycle. Consider having a 1-dimensional numpy array, `signal`, which is a neural signal time series sampled at 1000 Hz (`Fs`) that contains an alpha (8-12 Hz, `f_range`) oscillation. We can compute the table of cycle features with the following:

```python
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

signal = lowpass_filter(signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

Fs = 1000
f_range = (8, 12)
df = compute_features(signal, Fs, f_range)
```

Note that a lowpass filter is applied in order to remove high-frequency power that may interfere with extrema localization. (see section 0 of the [algorithm tutorial](https://github.com/bycycle-tools/bycycle/blob/master/tutorials/1_Cycle-by-cycle%20algorithm.ipynb) for more details).

It's necessary to note that the above `compute_features()` command used default parameters to localize extrema and detect bursts of oscillations. However, it is important to knowledgeably select these parameters, as described in the [algorithm tutorial](https://github.com/bycycle-tools/bycycle/blob/master/tutorials/1_Cycle-by-cycle%20algorithm.ipynb). The following example and text go over the different potential parameter changes:

```python

burst_kwargs = {'amplitude_fraction_threshold': .2,
                'amplitude_consistency_threshold': .5,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 3}

narrowband_kwargs = {'N_seconds': .5}

df = compute_features(signal, Fs, f_range,
                      center_extrema='T',
                      burst_detection_method='cycles',
                      burst_detection_kwargs=burst_kwargs,
                      find_extrema_kwargs={'filter_kwargs': narrowband_kwargs},
                      hilbert_increase_N=True)
```

* __center_extrema__ determines how the cycles are segmented. 'T' indicates the center extrema is a trough, so cycles are segmented peak-to-peak.
* __burst_detection_method__ selects which method for burst detection is used. The 'cycles' option uses features of adjacent cycles in order to detect bursts (e.g. period consistency, see next item). The 'amp' option uses an amplitude threshold to determine the cycles that are part of an oscillatory burst.
* __burst_detection_kwargs__ set the keyword arguments for the burst detection function. For the 'cycles' method, there are 5 keyword arguments (see [the end of the algorithm tutorial](https://github.com/bycycle-tools/bycycle/blob/master/tutorials/1_Cycle-by-cycle%20algorithm.ipynb) for advice on choosing these parameters).
* __find_extrema_kwargs__ set the keyword arguments for the function used to localize peaks and troughs. Most notably, you can change the duration of the bandpass filter (`N_seconds`) used during extrema localization (see section 1 of the [algorithm tutorial](https://github.com/bycycle-tools/bycycle/blob/master/tutorials/1_Cycle-by-cycle%20algorithm.ipynb)).
* __hilbert_increase_N__ is a boolean indicator of whether or not to zeropad the signal to bypass complications that `scipy.signal.hilbert()` has with some long signal durations. Try setting this parameter to `True` if this function is taking a long time to run. Note the Hilbert Transform is used to compute the `band_amp` feature of each cycle, which is the average analytic amplitude of the frequency of interest in that cycle. This is complementary to the `volt_amp` measure, and may be desired for some burst detection applications.

## Output

The output of `bycycle` is a pandas.DataFrame, a table like the one shown below (with many columns, so it is split into two images). Each row of this table corresponds to an individual segment of the signal, or a putative cycle of the rhythm of interest.

!["cycle dataframe part1"](img/cycledf_1.png)

!["cycle dataframe part2"](img/cycledf_2.png)

Some of the columns include:
* __sample_peak__ - the sample of the signal at which the peak of this cycle occurs
* __period__ - period of the cycle
* __time_peak__ - duration of the peak period
* __volt_amp__ - amplitude of this cycle, average of the rise and decay voltage
* __time_rdsym__ - rise-decay symmetry, the fraction of the cycle in the rise period (0.5 is symmetric)
* __time_ptsym__ - peak-trough symmetry, the fraction of the cycle in the peak period (0.5 is symmetric)
* __period_consistency__ - consistency between the periods of the adjacent cycles, used in burst detection
* __is_burst__ - indicator if the cycle is part of an oscillatory burst

The features in this table can then go on to be analyzed, as demonstrated in the [resting-state data tutorial](https://github.com/bycycle-tools/bycycle/blob/master/tutorials/2_Resting%20state%20cycle-by-cycle%20analysis.ipynb) and the [trial data tutorial](https://github.com/bycycle-tools/bycycle/blob/master/tutorials/3_Trial%20structure%20cycle-by-cycle%20analysis.ipynb). For example, we may be interested in the distribution of rise-decay symmetry values in a resting state recording, shown below.

!["rdsym distribution"](img/rdsym_distribution.png)

The plot below indicates in red the cycles of the signal that were identified as part of an oscillatory burst.

!["burst detection results"](img/bursts_detected.png)
