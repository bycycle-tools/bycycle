"""Tests plotting utilities."""

import numpy as np
import pandas as pd
from bycycle.utils import limit_df, limit_sig_times, get_extrema

###################################################################################################
###################################################################################################

def test_limit_df(sim_args):

    df = sim_args['df']
    sig = sim_args['sig']
    fs = sim_args['fs']

    xlims = (1, 2)

    df_short = limit_df(df, fs, xlims)

    assert df_short['sample_next_trough'].min() >= 0
    assert df_short['sample_last_trough'].max() <= fs * (xlims[1] - xlims[0])


def test_limit_sig_times(sim_args):

    df = sim_args['df']
    sig = sim_args['sig']
    fs = sim_args['fs']

    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlims = (1, 2)

    sig_short, times_short = limit_sig_times(sig, times, xlims)

    assert np.array_equal(times_short, times[fs*xlims[0]:fs*xlims[1]])
    assert np.array_equal(sig_short, sig[fs*xlims[0]:fs*xlims[1]])


def test_get_extrema(sim_args):

    df = sim_args['df']
    center_e, side_e = get_extrema(df)

    # The fixture will return peak centered cycles
    assert center_e == 'peak'
    assert side_e == 'trough'

    df = pd.DataFrame({'sample_trough': []})
    center_e, side_e = get_extrema(df)

    assert center_e == 'trough'
    assert side_e == 'peak'
