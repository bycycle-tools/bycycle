"""Tests for cyclepoints.zerox."""

import os

import numpy as np

from bycycle.cyclepoints import find_extrema

from bycycle.cyclepoints.zerox import *

###################################################################################################
###################################################################################################

def test_find_zerox(sim_stationary):
    """Test ability to find peaks and troughs."""

    # Load signal
    sig = sim_stationary

    fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    peaks, troughs = find_extrema(sig, fs, f_range, boundary=1, first_extrema='peak')

    # Find zerocrossings
    rises, decays = find_zerox(sig, peaks, troughs)

    assert len(peaks) == (len(rises) + 1)
    assert len(troughs) == len(decays)
    assert peaks[0] < decays[0]
    assert decays[0] < troughs[0]
    assert troughs[0] < rises[0]
    assert rises[0] < peaks[1]
