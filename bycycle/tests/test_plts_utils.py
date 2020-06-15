"""Tests plotting utilities."""

import numpy as np
import pandas as pd
from bycycle.plts.utils import apply_tlims, get_extrema

###################################################################################################
###################################################################################################

def test_apply_tlims(sim_args):
    """Test apply tlims to signal and dataframe."""

    df = sim_args['df']
    sig = sim_args['sig']
    fs = sim_args['fs']

    times = np.arange(0, len(sig) / fs, 1 / fs)
    tlims = (1, 2)

    df_short, sig_short, times_short = apply_tlims(df, sig, times, fs, tlims)

    print(df_short['sample_next_trough'].values)
    print(df['sample_next_trough'].values - int(fs * tlims[0]))

    assert np.array_equal(times_short, times[fs*tlims[0]:fs*tlims[1]])
    assert np.array_equal(sig_short,  sig[fs*tlims[0]:fs*tlims[1]])

    assert df_short['sample_next_trough'].min() >= 0
    assert df_short['sample_last_trough'].max() <= fs * (tlims[1] - tlims[0])


def test_get_extrema(sim_args):
    """Test determining extrema center/sides."""

    df = sim_args['df']
    center_e, side_e = get_extrema(df)

    # The fixture will return peak centered cycles
    assert center_e == 'peak'
    assert side_e == 'trough'

    df = pd.DataFrame({'sample_trough': []})
    center_e, side_e = get_extrema(df)

    assert center_e == 'trough'
    assert side_e == 'peak'

