"""Tests the functions to identify points in cycles work

NOTES
-----
The tests here are not strong tests for accuracy.
    They serve rather as 'smoke tests', for if anything fails completely.
"""

from bycycle import cyclepoints
import numpy as np
from scipy.signal import argrelextrema
import os

# Set data path
data_path = '/'.join(os.path.dirname(bycycle.__file__).split('/')[:-1]) + '/tutorials/data/'


def test_find_extrema():
    """Test ability to find peaks and troughs"""

    # Load signal
    signal = np.load(data_path + 'sim_stationary.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

    # find local maxima and minima using scipy
    maxima = argrelextrema(signal, np.greater)
    minima = argrelextrema(signal, np.less)

    # Find peaks and troughs using bycycle and make sure match scipy
    f_range = (6, 14)
    Ps, Ts = cyclepoints.find_extrema(signal, Fs, f_range, boundary=1,
                                      first_extrema='trough')
    assert len(Ps) == len(Ts)
    assert Ts[0] < Ps[0]
    np.testing.assert_allclose(Ps, maxima[0])
    np.testing.assert_allclose(Ts[:len(Ps)], minima[0][:len(Ps)])

    # Test first extrema again
    Ps, Ts = cyclepoints.find_extrema(signal, Fs, f_range, boundary=1,
                                      first_extrema='peak')
    assert Ps[0] < Ts[0]


def test_find_zerox():
    """Test ability to find peaks and troughs"""

    # Load signal
    signal = np.load(data_path + 'sim_stationary.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

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
    """Test waveform phase estimate"""

    # Load signal
    signal = np.load(data_path + 'sim_stationary.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

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
