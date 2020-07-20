"""Tests the functions to identify points in cycles work."""

import os
import numpy as np
from scipy.signal import argrelextrema
import pytest

from bycycle.cyclepoints import find_extrema, find_zerox, extrema_interpolated_phase

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


def test_find_zerox():
    """Test ability to find peaks and troughs."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

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


def test_extrema_interpolated_phase():
    """Test waveform phase estimate."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    peaks, troughs = find_extrema(sig, fs, f_range, boundary=1, first_extrema='peak')

    # Find zerocrossings
    rises, decays = find_zerox(sig, peaks, troughs)

    # Compute phase
    pha = extrema_interpolated_phase(sig, peaks, troughs, rises=rises, decays=decays)

    assert len(pha) == len(sig)
    assert np.all(np.isclose(pha[peaks], 0))
    assert np.all(np.isclose(pha[troughs], -np.pi))
    assert np.all(np.isclose(pha[rises], -np.pi/2))
    assert np.all(np.isclose(pha[decays], np.pi/2))
