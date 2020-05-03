"""Tests the functions to identify points in cycles work."""

import bycycle
from bycycle import cyclepoints
import numpy as np
from scipy.signal import argrelextrema
import os
import pytest

# Set data path
DATA_PATH = '/'.join(os.path.dirname(bycycle.__file__).split('/')[:-1]) + '/tutorials/data/'

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
    signal = np.load(DATA_PATH + 'sim_stationary.npy')
    Fs = 1000
    f_range = (6, 14)

    # find local maxima and minima using scipy
    maxima = argrelextrema(signal, np.greater)
    minima = argrelextrema(signal, np.less)

    # Find peaks and troughs using bycycle and make sure match scipy
    f_range = (6, 14)
    Ps, Ts = cyclepoints.find_extrema(signal, Fs, f_range, boundary=1,
                                      first_extrema=first_extrema)
    if first_extrema == 'trough':
        assert len(Ps) == len(Ts)
        assert Ts[0] < Ps[0]
        np.testing.assert_allclose(Ps, maxima[0])
        np.testing.assert_allclose(Ts[:len(Ps)], minima[0][:len(Ps)])
    elif first_extrema == 'peak':
        assert Ps[0] < Ts[0]


def test_find_zerox():
    """Test ability to find peaks and troughs."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_stationary.npy')
    Fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    Ps, Ts = cyclepoints.find_extrema(signal, Fs, f_range, boundary=1,
                                      first_extrema='peak')

    # Find zerocrossings
    zeroxR, zeroxD = cyclepoints.find_zerox(signal, Ps, Ts)
    assert len(Ps) == (len(zeroxR) + 1)
    assert len(Ts) == len(zeroxD)
    assert Ps[0] < zeroxD[0]
    assert zeroxD[0] < Ts[0]
    assert Ts[0] < zeroxR[0]
    assert zeroxR[0] < Ps[1]


def test_extrema_interpolated_phase():
    """Test waveform phase estimate."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_stationary.npy')
    Fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    Ps, Ts = cyclepoints.find_extrema(signal, Fs, f_range, boundary=1,
                                      first_extrema='peak')

    # Find zerocrossings
    zeroxR, zeroxD = cyclepoints.find_zerox(signal, Ps, Ts)

    # Compute phase
    pha = cyclepoints.extrema_interpolated_phase(signal, Ps, Ts, zeroxR=zeroxR, zeroxD=zeroxD)
    assert len(pha) == len(signal)
    assert np.all(np.isclose(pha[Ps], 0))
    assert np.all(np.isclose(pha[Ts], -np.pi))
    assert np.all(np.isclose(pha[zeroxR], -np.pi/2))
    assert np.all(np.isclose(pha[zeroxD], np.pi/2))
