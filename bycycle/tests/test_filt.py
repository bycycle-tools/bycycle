"""Tests the filtering methods work

NOTES
-----
The tests here are not strong tests for accuracy.
    They serve rather as 'smoke tests', for if anything fails completely.
"""

from bycycle import filt
import numpy as np

def test_bandpass():
    """Test bandpass filter functionality"""

    # Simulate fake data
    np.random.seed(0)
    cf = 10 # Oscillation center frequency
    T = 10 # Recording duration (seconds)
    Fs = 1000 # Sampling rate
    signal = sim.sim_noisy_bursty_oscillator(cf, T, Fs, prob_enter_burst=.1,
                                             prob_leave_burst=.1, SNR=5, rdsym=.2)

    # Test output same length as input
    N_seconds = 0.5
    signal_filt = filt.bandpass_filter(signal, Fs, (8, 12),
                                       N_seconds=N_seconds)
    assert len(signal) == len(signal_filt)

    # Test edge artifacts removed appropriately
    N_samples_filter = int(np.ceil(Fs * N_seconds))
    if N_samples_filter % 2 == 0:
        N_samples_filter = int(N_samples_filter + 1)
    N_samples_NaN = int(np.ceil(N_samples_filter / 2))
    assert np.all(np.isnan(signal_filt[:N_samples_NaN]))
    assert np.all(np.isnan(signal_filt[-N_samples_NaN:]))
    assert np.all(np.logical_not(np.isnan(signal_filt[N_samples_NaN:-N_samples_NaN])))

    # Test edge artifacts are not removed if desired
    signal_filt = filt.bandpass_filter(signal, Fs, (8, 12),
                                       N_seconds=N_seconds,
                                       remove_edge_artifacts=False)
    assert np.all(np.logical_not(np.isnan(signal_filt)))

    # Test to get warning if transition band is too wide
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        signal_filt = filt.bandpass_filter(signal, Fs, (8, 12),
                                       	   N_seconds=.1)
        # Verify some things
        assert len(w) == 1
        assert "Filter bandwidth" in str(w[-1].message)

    # Test returns kernel and signal
    out = filt.bandpass_filter(signal, Fs, (8, 12), N_seconds=N_seconds,
                               return_kernel=True)
    assert len(out) == 2

    # Test same result if N_cycle and N_seconds used
    filt1 = filt.bandpass_filter(signal, Fs, (8, 12), N_seconds=1,
                                 remove_edge_artifacts=False)
    filt2 = filt.bandpass_filter(signal, Fs, (8, 12), N_cycles=8,
                                 remove_edge_artifacts=False)
    np.testing.assert_allclose(filt1, filt2)
