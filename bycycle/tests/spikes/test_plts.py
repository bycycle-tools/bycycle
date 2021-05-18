"""Tests for plotting spikes."""

import pytest

from neurodsp.tests.tutils import plot_test

from bycycle.spikes.plts import plot_spikes, _infer_labels

###################################################################################################
###################################################################################################

@plot_test
@pytest.mark.parametrize('index', [None, 0])
@pytest.mark.parametrize('spikes_arg', [True, False])
@pytest.mark.parametrize('xlim', [None, (0, .1)])
def test_plot_spikes(sim_spikes, sim_spikes_fit, index, spikes_arg, xlim):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']

    spikes = sim_spikes_fit['spikes']
    df_features = spikes.df_features.copy()

    if spikes_arg:
        plot_spikes(df_features, sig, fs, spikes=spikes, index=index, xlim=xlim)
    else:
        plot_spikes(df_features, sig, fs, spikes=None, index=index, xlim=xlim)


@pytest.mark.parametrize('center_e', ['trough', 'peak'])
def test_infer_labels(center_e):

    keys, labels = _infer_labels(center_e)

    if center_e == 'trough':
        assert keys[0] == 'Trough'
        assert labels[0] == 'sample_trough'
    else:
        assert keys[0] == 'Peak'
        assert labels[0] == 'sample_peak'
