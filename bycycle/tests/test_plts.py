"""Tests plotting."""

import pytest

from neurodsp.sim import sim_oscillation
from bycycle.features import compute_features
from bycycle.plts import plot_burst_detect_params

@pytest.mark.parametrize("only_result", [True, False])
def test_plot_burst_detect_params(only_result):
    """Test plotting burst detection."""

    # Simulate oscillating time series
    n_seconds = 25
    fs = 1000
    freq = 10
    f_range = (6, 14)

    osc_kwargs = {'amplitude_fraction_threshold': 0,
                  'amplitude_consistency_threshold': .5,
                  'period_consistency_threshold': .5,
                  'monotonicity_threshold': .8,
                  'n_cycles_min': 3}

    sig = sim_oscillation(n_seconds, fs, freq)

    df = compute_features(sig, fs, f_range)

    fig = plot_burst_detect_params(sig, fs, df, osc_kwargs, plot_only_result=only_result)

    if not only_result:
        for param in fig:
            assert param is not None
    else:
        assert fig is not None
