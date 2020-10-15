"""Tests for cyclepoints.extrema."""

import os

import numpy as np
from scipy.signal import argrelextrema

import pytest

from bycycle.cyclepoints.extrema import *
from bycycle.tests.settings import FS, F_RANGE

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
def test_find_extrema(sim_args, first_extrema):
    """Test ability to find peaks and troughs."""

    # Load signal
    sig = sim_args['sig']

    # find local maxima and minima using scipy
    maxima = argrelextrema(sig, np.greater)
    minima = argrelextrema(sig, np.less)

    # Find peaks and troughs using bycycle and make sure match scipy
    peaks, troughs = find_extrema(sig, FS, F_RANGE, first_extrema=first_extrema)

    if first_extrema == 'trough':
        assert troughs[0] < peaks[0]
        assert len(peaks) == len(troughs)

        # The first extrema is actually a peak; slice maxima to account for this when
        #   trough is forced as the first extrema.
        np.testing.assert_equal(peaks, maxima[0][1:])

        # When first_extrema is trough, the last extrema must be a peak. This leads to a missing
        #   last trough.
        np.testing.assert_equal(troughs, minima[0][:-1])

    elif first_extrema == 'peak':
        assert peaks[0] < troughs[0]
        assert len(peaks) == len(troughs)
        np.testing.assert_equal(peaks, maxima[0])
        np.testing.assert_equal(troughs, minima[0])




