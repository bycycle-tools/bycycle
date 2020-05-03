"""Tests burst detection."""

import bycycle
import numpy as np
from bycycle import burst, filt, features, sim
import itertools
import os
import pytest

# Set data path
DATA_PATH = '/'.join(os.path.dirname(bycycle.__file__).split('/')[:-1]) + '/tutorials/data/'

###################################################################################################
###################################################################################################

def test_detect_bursts_cycles():
    """Test amplitude and period consistency burst detection."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_bursting.npy')
    Fs = 1000
    f_range = (6, 14)

    signal = filt.lowpass_filter(signal, Fs, 30, N_seconds=.3,
                                 remove_edge_artifacts=False)

    # Compute cycle-by-cycle df without burst detection column
    df = features.compute_features(signal, Fs, f_range,
                                   burst_detection_method='amp',
                                   burst_detection_kwargs={'amp_threshes': (1, 2),
                                                           'filter_kwargs': {'N_seconds': .5}})
    df.drop('is_burst', axis=1, inplace=True)

    # Apply consistency burst detection
    df_burst_cycles = burst.detect_bursts_cycles(df, signal)

    # Make sure that burst detection is only boolean
    assert df_burst_cycles.dtypes['is_burst'] == 'bool'
    assert df_burst_cycles['is_burst'].mean() > 0
    assert df_burst_cycles['is_burst'].mean() < 1
    assert np.min([sum(1 for _ in group) for key, group in \
        itertools.groupby(df_burst_cycles['is_burst']) if key]) >= 3


def test_detect_bursts_df_amp():
    """Test amplitude-threshold burst detection."""

    # Load signal
    signal = np.load(DATA_PATH + 'sim_bursting.npy')
    Fs = 1000
    f_range = (6, 14)
    signal = filt.lowpass_filter(signal, Fs, 30, N_seconds=.3,
                                 remove_edge_artifacts=False)

    # Compute cycle-by-cycle df without burst detection column
    df = features.compute_features(signal, Fs, f_range,
                                   burst_detection_method='amp',
                                   burst_detection_kwargs={'amp_threshes': (1, 2),
                                                           'filter_kwargs': {'N_seconds': .5}})
    df.drop('is_burst', axis=1, inplace=True)

    # Apply consistency burst detection
    df_burst_amp = burst.detect_bursts_df_amp(df, signal, Fs, f_range,
                                              amp_threshes=(.5, 1),
                                              N_cycles_min=4, filter_kwargs={'N_seconds': .5})

    # Make sure that burst detection is only boolean
    assert df_burst_amp.dtypes['is_burst'] == 'bool'
    assert df_burst_amp['is_burst'].mean() > 0
    assert df_burst_amp['is_burst'].mean() < 1
    assert np.min([sum(1 for _ in group) for key, group \
        in itertools.groupby(df_burst_amp['is_burst']) if key]) >= 4


@pytest.mark.parametrize("only_result", [True, False])
def test_plot_burst_detect_params(only_result):
    """Test plotting burst detection."""

    # Simulate oscillating time series
    T = 25
    Fs = 1000
    freq = 10
    f_range = (6, 14)
    osc_kwargs = {'amplitude_fraction_threshold': 0,
                  'amplitude_consistency_threshold': .5,
                  'period_consistency_threshold': .5,
                  'monotonicity_threshold': .8,
                  'N_cycles_min': 3}
    x = sim.sim_oscillator(T, Fs, freq)

    df = features.compute_features(x, Fs, f_range)
    fig = burst.plot_burst_detect_params(x, Fs, df, osc_kwargs,
                                         plot_only_result=only_result)

    if not only_result:
        for param in fig:
            assert param is not None
    else:
        assert fig is not None


@pytest.mark.parametrize("amp_threshes",
    [
        (1, 2),
        pytest.param((1, 2, 3), marks=pytest.mark.xfail(raises=ValueError))
    ]
)
@pytest.mark.parametrize("magnitude_type",
    [
        'amplitude',
        'power',
        pytest.param('fail', marks=pytest.mark.xfail(raises=ValueError))
    ]
)
@pytest.mark.parametrize("return_amplitude", [True, False])
def test_twothresh_amp(amp_threshes, magnitude_type, return_amplitude):
    """Burst detection using two amplitude thresholds."""

    T = 25
    Fs = 1000
    freq = 10
    f_range = (6, 14)
    if magnitude_type == 'power':
        amp_threshes = tuple((thr**(1/2) for thr in amp_threshes))

    x = sim.sim_oscillator(T, Fs, freq)

    if return_amplitude:
        isosc_noshort, x_magnitude = \
            burst.twothresh_amp(x, Fs, f_range, amp_threshes, N_cycles_min=3,
                                magnitude_type=magnitude_type,
                                return_amplitude=return_amplitude,
                                filter_kwargs=None)
        assert len(x_magnitude) == T * Fs
    else:
        isosc_noshort = \
            burst.twothresh_amp(x, Fs, f_range, amp_threshes, N_cycles_min=3,
                                magnitude_type=magnitude_type,
                                return_amplitude=return_amplitude,
                                filter_kwargs=None)

    assert len(isosc_noshort) == T * Fs
