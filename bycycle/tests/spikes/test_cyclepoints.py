"""Test for spike cyclepoints."""

import numpy as np

from bycycle.spikes.cyclepoints import compute_spike_cyclepoints

from pytest import raises

###################################################################################################
###################################################################################################


def test_compute_spike_cyclepoints(sim_spikes):

    # Unpack pytest fixture
    sig = sim_spikes['sig']
    spike = sim_spikes['spike']
    fs = sim_spikes['fs']
    locs = sim_spikes['locs']
    starts = locs[0]
    ends = locs[1]

    # Case0: 3-current spikes
    f_range = (500, 3000)
    df = compute_spike_cyclepoints(sig, fs, f_range, std=2)

    assert len(df) == 5

    spike_trough_idx = np.argmin(spike)
    spike_last_peak_idx = np.argmax(spike[:spike_trough_idx])
    spike_next_peak_idx = np.argmax(spike[spike_trough_idx:]) + spike_trough_idx

    assert (starts + spike_trough_idx == df['sample_trough']).all()
    assert (starts + spike_last_peak_idx == df['sample_last_peak']).all()
    assert (starts + spike_next_peak_idx == df['sample_next_peak']).all()

    # Case1: Std too low
    with raises(ValueError):
        df = compute_spike_cyclepoints(sig, fs, f_range, std=10000)

    # Case2: Contains overlapping spikes
    sig_overlap = sim_spikes['sig_overlap']
    df = compute_spike_cyclepoints(sig_overlap, fs, f_range, std=2)
