"""Tests plotting utilities."""

import numpy as np
import pandas as pd

from bycycle.utils import limit_df, limit_signal, get_extrema_df

###################################################################################################
###################################################################################################

def test_limit_df(sim_args):

    df_samples = sim_args['df_samples']
    fs = sim_args['fs']

    xlim = (1, 2)

    df_short = limit_df(df_samples, fs, start=xlim[0], stop=xlim[1])

    assert df_short['sample_next_trough'].min() >= 0
    assert df_short['sample_last_trough'].max() <= fs * (xlim[1] - xlim[0])


def test_limit_signal(sim_args):

    sig = sim_args['sig']
    fs = sim_args['fs']

    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlim = (1, 2)

    sig_short, times_short = limit_signal(times, sig, start=xlim[0], stop=xlim[1])

    assert np.array_equal(times_short, times[fs*xlim[0]:fs*xlim[1]])
    assert np.array_equal(sig_short, sig[fs*xlim[0]:fs*xlim[1]])


def test_get_extrema_df(sim_args):

    df_samples = sim_args['df_samples']
    center_e, side_e = get_extrema_df(df_samples)

    # The fixture will return peak centered cycles
    assert center_e == 'peak'
    assert side_e == 'trough'

    df_samples = pd.DataFrame({'sample_trough': []})
    center_e, side_e = get_extrema_df(df_samples)

    assert center_e == 'trough'
    assert side_e == 'peak'
