"""Tests plotting."""

import pytest
from neurodsp.sim import sim_oscillation
from bycycle.features import compute_features
from bycycle.plts import plot_burst_detect_params, plot_cyclepoints
from bycycle.cyclepoints import find_zerox, find_extrema

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


@pytest.mark.parametrize("plot_type",
    [
            'extrema',
            'zerox',
            'both',
            pytest.param(None, marks=pytest.mark.xfail(raises=TypeError))
    ]
)
def test_plot_cyclepoints(plot_type):
    """Test plotting cyclepoints."""

    # Simulate oscillating time series
    n_seconds = 10
    fs = 500
    freq = 10

    sig = sim_oscillation(n_seconds, fs, freq)

    # Find extrema and zero-crossings
    ps, ts = find_extrema(sig, fs, (8, 12))
    zerox_rise, zerox_decay = find_zerox(sig, ps, ts)

    if plot_type == 'both':
        ax = plot_cyclepoints(sig, fs, extrema=(ps, ts), zerox=(zerox_rise, zerox_decay))
    elif plot_type == 'extrema':
        ax = plot_cyclepoints(sig, fs, extrema=(ps, ts))
    elif plot_type == 'zerox':
        ax = plot_cyclepoints(sig, fs, zerox=(zerox_rise, zerox_decay))
    else:
        ax = plot_cyclepoints(sig, fs)

    assert ax is not None



