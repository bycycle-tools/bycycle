"""Tests for cyclepoints.extrema."""

import os

import numpy as np
from scipy.signal import argrelextrema

import pytest

from bycycle.cyclepoints.extrema import *

# Set data path
DATA_PATH = os.getcwd() + '/tutorials/data/'

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("first_extrema",
    [
        'peak',
        'trough',
        None,
        pytest.param('fail', marks=pytest.mark.xfail(raises=ValueError))
    ]
)
def test_find_extrema(first_extrema):
    """Test ability to find peaks and troughs."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # find local maxima and minima using scipy
    maxima = argrelextrema(sig, np.greater)
    minima = argrelextrema(sig, np.less)

    # Find peaks and troughs using bycycle and make sure match scipy
    peaks, troughs = find_extrema(sig, fs, f_range, boundary=1, first_extrema=first_extrema)

    if first_extrema == 'trough':
        assert len(peaks) == len(troughs)
        assert troughs[0] < peaks[0]
        np.testing.assert_allclose(peaks, maxima[0])
        np.testing.assert_allclose(troughs[:len(peaks)], minima[0][:len(peaks)])
    elif first_extrema == 'peak':
        assert peaks[0] < troughs[0]
