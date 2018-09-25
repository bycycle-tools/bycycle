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

    # Load signal
    signal = np.load('data/sim_bursting.npy')
    Fs = 1000  # Sampling rate

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
    assert np.all(np.logical_not(np.isnan(
        signal_filt[N_samples_NaN:-N_samples_NaN])))

    # Test edge artifacts are not removed if desired
    signal_filt = filt.bandpass_filter(signal, Fs, (8, 12),
                                       N_seconds=N_seconds,
                                       remove_edge_artifacts=False)
    assert np.all(np.logical_not(np.isnan(signal_filt)))

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


def test_lowpass():
    """Test lowpass filter functionality"""

    # Load signal
    signal = np.load('data/sim_bursting.npy')
    Fs = 1000  # Sampling rate

    # Test output same length as input
    N_seconds = 0.5
    signal_filt = filt.lowpass_filter(signal, Fs, 30,
                                      N_seconds=N_seconds)
    assert len(signal) == len(signal_filt)

    # Test edge artifacts removed appropriately
    N_samples_filter = int(np.ceil(Fs * N_seconds))
    if N_samples_filter % 2 == 0:
        N_samples_filter = int(N_samples_filter + 1)
    N_samples_NaN = int(np.ceil(N_samples_filter / 2))
    assert np.all(np.isnan(signal_filt[:N_samples_NaN]))
    assert np.all(np.isnan(signal_filt[-N_samples_NaN:]))
    assert np.all(np.logical_not(np.isnan(
        signal_filt[N_samples_NaN:-N_samples_NaN])))

    # Test edge artifacts are not removed if desired
    signal_filt = filt.bandpass_filter(signal, Fs, (8, 12),
                                       N_seconds=N_seconds,
                                       remove_edge_artifacts=False)
    assert np.all(np.logical_not(np.isnan(signal_filt)))


def test_amp():
    """Test phase time series functionality"""

    # Load signal
    signal = np.load('data/sim_bursting.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

    # Test output same length as input
    amp = filt.amp_by_time(signal, Fs, f_range, filter_kwargs={'N_seconds': .5})
    assert len(signal) == len(amp)

    # Test results are the same if add NaNs to the side
    signal_nan = np.pad(signal, 10,
                        mode='constant',
                        constant_values=(np.nan,))
    amp_nan = filt.amp_by_time(signal_nan, Fs, f_range, filter_kwargs={'N_seconds': .5})
    np.testing.assert_allclose(amp_nan[10:-10], amp)

    # Test NaN is in same places as filtered signal
    signal_filt = filt.bandpass_filter(signal, Fs, (6, 14), N_seconds=.5)
    assert np.all(np.logical_not(
                  np.logical_xor(np.isnan(amp), np.isnan(signal_filt))))

    # Test works fine if input signal already has NaN
    signal_low = filt.lowpass_filter(signal, Fs, 30, N_seconds=.3)
    amp = filt.amp_by_time(signal_low, Fs, f_range,
                           filter_kwargs={'N_seconds': .5})
    assert len(signal) == len(amp)

    # Test option to not remove edge artifacts
    amp = filt.amp_by_time(signal, Fs, f_range,
                           filter_kwargs={'N_seconds': .5},
                           remove_edge_artifacts=False)
    assert np.all(np.logical_not(np.isnan(amp)))


def test_phase():
    """Test phase time series functionality"""

    # Load signal
    signal = np.load('data/sim_bursting.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

    # Test output same length as input
    pha = filt.phase_by_time(signal, Fs, f_range, filter_kwargs={'N_seconds': .5})
    assert len(signal) == len(pha)

    # Test results are the same if add NaNs to the side
    signal_nan = np.pad(signal, 10,
                        mode='constant',
                        constant_values=(np.nan,))
    pha_nan = filt.phase_by_time(signal_nan, Fs, f_range, filter_kwargs={'N_seconds': .5})
    np.testing.assert_allclose(pha_nan[10:-10], pha)

    # Test NaN is in same places as filtered signal
    signal_filt = filt.bandpass_filter(signal, Fs, (6, 14), N_seconds=.5)
    assert np.all(np.logical_not(
                  np.logical_xor(np.isnan(pha), np.isnan(signal_filt))))

    # Test works fine if input signal already has NaN
    signal_low = filt.lowpass_filter(signal, Fs, 30, N_seconds=.3)
    pha = filt.phase_by_time(signal_low, Fs, f_range,
                             filter_kwargs={'N_seconds': .5})
    assert len(signal) == len(pha)

    # Test option to not remove edge artifacts
    pha = filt.phase_by_time(signal, Fs, f_range,
                             filter_kwargs={'N_seconds': .5},
                             remove_edge_artifacts=False)
    assert np.all(np.logical_not(np.isnan(pha)))
