"""Tests the neural signal simulation functionality

NOTES
-----
The tests here are not strong tests for accuracy.
    They serve rather as 'smoke tests', for if anything fails completely.
"""

import numpy as np
import pandas as pd
from bycycle import sim


def test_sim_filtered_brown_noise():
    """Test brown noise simulation"""
    np.random.seed(0)
    T = 5
    Fs = 1000
    f_range = (2, None)
    N = 1001
    x = sim.sim_filtered_brown_noise(T, Fs, f_range, N)
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0


def test_sim_oscillator():
    """Test stationary oscillation simulation"""
    np.random.seed(0)
    T = 5
    Fs = 1000
    freq = 10
    x = sim.sim_oscillator(T, Fs, freq)
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0

    # Make sure that it generated a matching cosine
    cosine = np.cos(np.arange(len(x)) * freq / Fs * 2 * np.pi)
    np.testing.assert_allclose(x, cosine, atol=10e-7)


def test_sim_noisy_oscillator():
    """Test noisy stationary oscillation simulation"""
    np.random.seed(0)
    T = 5
    Fs = 1000
    freq = 10
    x = sim.sim_noisy_oscillator(T, Fs, freq, SNR=5)
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0


def test_sim_bursty_oscillator():
    """Test bursting oscillation simulation"""
    np.random.seed(0)
    T = 5
    Fs = 1000
    freq = 10
    x = sim.sim_bursty_oscillator(T, Fs, freq, prob_enter_burst=.1,
                                  prob_leave_burst=.1,
                                  cycle_features={'rdsym_mean': .3,
                                                  'rdsym_std': .05,
                                                  'rdsym_burst_std': 0})
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0


def test_sim_noisy_bursty_oscillator():
    """Test noisy bursting oscillation simulation"""
    T = 10
    Fs = 1000
    freq = 10
    x = sim.sim_noisy_bursty_oscillator(T, Fs, freq,
                                        cycle_features={'rdsym_mean': .3,
                                                        'rdsym_std': .05,
                                                        'rdsym_burst_std': 0})
    assert len(x) == int(T * Fs)
    assert np.isnan(x).sum() == 0

    signal, oscillator, brown, df = sim.sim_noisy_bursty_oscillator(T, Fs, freq, return_components=True, return_cycle_df=True)
    assert len(signal) == int(T * Fs)
    assert np.isnan(signal).sum() == 0
    assert len(oscillator) == int(T * Fs)
    assert np.isnan(oscillator).sum() == 0
    assert len(brown) == int(T * Fs)
    assert np.isnan(brown).sum() == 0
    assert type(df) == pd.core.frame.DataFrame
