"""Tests the neural signal simulation functionality

NOTES
-----
The tests here are not strong tests for accuracy.
    They serve rather as 'smoke tests', for if anything fails completely.
"""

import numpy as np
from bycycle import sim


def test_sim_filtered_brown_noise():
    """Test brown noise simulation"""
    T = 5
    Fs = 1000
    f_range = (2, None)
    N = 1001
    x = sim.sim_filtered_brown_noise(T, Fs, f_range, N)
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0


def test_sim_oscillator():
    """Test stationary oscillation simulation"""
    T = 5
    Fs = 1000
    freq = 10
    x = sim.sim_oscillator(T, Fs, freq)
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0

    # Make sure that it generated a matching cosine
    cosine = np.cos(np.arange(len(x)) * freq / Fs * 2 * np.pi)
    np.testing.assert_allclose(x, cosine, atol=10e-7)
