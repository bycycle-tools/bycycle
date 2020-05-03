"""Test filtering."""

import bycycle
from bycycle import filt
import numpy as np
import os
import pytest

# Set data path
DATA_PATH = '/'.join(os.path.dirname(bycycle.__file__).split('/')[:-1]) + '/tutorials/data/'

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("fc_sigerr",
    [
        [(8, 12), False],
        pytest.param([(12, 8, 10), False],
                     marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param([(8, 12), True],
                     marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param([(12, 8), False],
                     marks=pytest.mark.xfail(raises=ValueError))
    ]
)
def test_bandpass(fc_sigerr):
    """Test bandpass filter."""
    fc = fc_sigerr[0]
    sig_err = fc_sigerr[1]

    # Load signal
    signal = np.load(DATA_PATH + 'sim_bursting.npy')
    if sig_err:
      signal = signal[:5]

    Fs = 1000

    # Test output same length as input
    N_seconds = 0.5
    signal_filt = filt.bandpass_filter(signal, Fs, fc,
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
    signal_filt = filt.bandpass_filter(signal, Fs, fc,
                                       N_seconds=N_seconds,
                                       remove_edge_artifacts=False,
                                       plot_frequency_response=True,
                                       print_transition_band=True)
    assert np.all(np.logical_not(np.isnan(signal_filt)))

    # Test returns kernel and signal
    out = filt.bandpass_filter(signal, Fs, fc, N_seconds=N_seconds,
                               return_kernel=True)
    assert len(out) == 2

    # Test same result if N_cycle and N_seconds used
    filt1 = filt.bandpass_filter(signal, Fs, fc, N_seconds=1,
                                 remove_edge_artifacts=False)
    filt2 = filt.bandpass_filter(signal, Fs, fc, N_cycles=8,
                                 remove_edge_artifacts=False)
    np.testing.assert_allclose(filt1, filt2)


def test_lowpass():
    """Test lowpass filter."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_bursting.npy')
    Fs = 1000

    # Test output same length as input
    N_seconds = 0.5
    signal_filt, _ = filt.lowpass_filter(signal, Fs, 30, return_kernel=True)
    signal_filt = filt.lowpass_filter(signal, Fs, 30,
                                      N_seconds=N_seconds,
                                      plot_frequency_response=True)
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
    """Test phase time series."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_bursting.npy')
    Fs = 1000
    f_range = (6, 14)

    # Test output same length as input
    amp = filt.amp_by_time(signal, Fs, f_range)
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
    """Test phase time series."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_bursting.npy')
    Fs = 1000
    f_range = (6, 14)

    # Test output same length as input
    pha = filt.phase_by_time(signal, Fs, f_range, hilbert_increase_N=True)
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
