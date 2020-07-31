"""Tests for features.cyclepoints."""

from bycycle.cyclepoints import find_extrema, find_zerox

from bycycle.features.cyclepoints import *

###################################################################################################
###################################################################################################

def test_compute_cyclepoints(sim_args):

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    peaks, troughs = find_extrema(sig, fs, f_range)
    rises, decays = find_zerox(sig, peaks, troughs)

    df_samples = compute_cyclepoints(sig, fs, f_range)

    assert (df_samples['sample_peak'] == peaks[1:]).all()
    assert (df_samples['sample_zerox_decay'] == decays[1:]).all()
    assert (df_samples['sample_zerox_rise'] == rises).all()
    assert (df_samples['sample_last_trough'] == troughs[:-1]).all()
    assert (df_samples['sample_next_trough'] == troughs[1:]).all()
