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
    ps, ts = find_extrema(sig, fs, f_range, boundary=1, first_extrema=first_extrema)

    if first_extrema == 'trough':
        assert len(ps) == len(ts)
        assert ts[0] < ps[0]
        np.testing.assert_allclose(ps, maxima[0])
        np.testing.assert_allclose(ts[:len(ps)], minima[0][:len(ps)])
    elif first_extrema == 'peak':
        assert ps[0] < ts[0]


def test_find_zerox():
    """Test ability to find peaks and troughs."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    ps, ts = find_extrema(sig, fs, f_range, boundary=1, first_extrema='peak')

    # Find zerocrossings
    zerox_rise, zerox_decay = find_zerox(sig, ps, ts)

    assert len(ps) == (len(zerox_rise) + 1)
    assert len(ts) == len(zerox_decay)
    assert ps[0] < zerox_decay[0]
    assert zerox_decay[0] < ts[0]
    assert ts[0] < zerox_rise[0]
    assert zerox_rise[0] < ps[1]


def test_extrema_interpolated_phase():
    """Test waveform phase estimate."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    ps, ts = find_extrema(sig, fs, f_range, boundary=1, first_extrema='peak')

    # Find zerocrossings
    zerox_rise, zerox_decay = find_zerox(sig, ps, ts)

    # Compute phase
    pha = extrema_interpolated_phase(sig, ps, ts, zerox_rise=zerox_rise, zerox_decay=zerox_decay)

    assert len(pha) == len(sig)
    assert np.all(np.isclose(pha[ps], 0))
    assert np.all(np.isclose(pha[ts], -np.pi))
    assert np.all(np.isclose(pha[zerox_rise], -np.pi/2))
    assert np.all(np.isclose(pha[zerox_decay], np.pi/2))
